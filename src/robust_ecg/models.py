"""xresnet1d50 as in the PTB-XL benchmark (Strodthoff et al., code/models/xresnet1d.py).

Faithful port without the fastai dependency: stem (32, 32, 64), kernel size 5, bottleneck blocks
[3, 4, 6, 3] with expansion 4 and a constant width of 64 in every stage (so ~1M parameters, not the
ImageNet widths), concat pooling head with one hidden layer of 128 units and dropout (0.25, 0.5).
The model is split into `encoder` (for SSL pretraining) and `head`.
"""
import torch
import torch.nn as nn


def conv_bn(ni, nf, ks, stride=1, act=True, zero_bn=False):
    conv = nn.Conv1d(ni, nf, ks, stride=stride, padding=(ks - 1) // 2, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    nn.init.constant_(bn.bias, 1e-3)
    return nn.Sequential(conv, bn, nn.ReLU(inplace=True)) if act else nn.Sequential(conv, bn)


class Bottleneck(nn.Module):
    def __init__(self, ni, nh, nf, stride, ks):
        super().__init__()
        self.convs = nn.Sequential(conv_bn(ni, nh, 1), conv_bn(nh, nh, ks, stride=stride),
                                   conv_bn(nh, nf, 1, act=False, zero_bn=True))
        idp = []
        if stride != 1:
            idp.append(nn.AvgPool1d(2, ceil_mode=True))
        if ni != nf:
            idp.append(conv_bn(ni, nf, 1, act=False))
        self.idpath = nn.Sequential(*idp)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convs(x) + self.idpath(x))


class Encoder(nn.Module):
    """Input (B, 12, T) -> features (B, 64*4, T') before pooling."""

    def __init__(self, in_ch=12, layers=(3, 4, 6, 3), width=64, expansion=4, ks=5, stem=(32, 32, 64)):
        super().__init__()
        szs = [in_ch, *stem]
        mods = [conv_bn(szs[i], szs[i + 1], ks, stride=2 if i == 0 else 1) for i in range(3)]
        mods.append(nn.MaxPool1d(3, stride=2, padding=1))
        ni = stem[-1]                            # 64 channels out of the stem (= 64//4 * 4)
        for i, n in enumerate(layers):
            for j in range(n):
                mods.append(Bottleneck(ni, width, width * expansion, stride=2 if (i > 0 and j == 0) else 1, ks=ks))
                ni = width * expansion
        self.body = nn.Sequential(*mods)
        self.out_dim = ni

    def forward(self, x):
        return self.body(x)


class ConcatPool(nn.Module):
    def forward(self, x):
        return torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)


def make_head(nf, nc, hidden=128, ps=(0.25, 0.5)):
    """fastai create_head1d(lin_ftrs=[128], ps=0.5): pool -> [BN, Dropout, Linear, ReLU] -> [BN, Dropout, Linear]."""
    head = nn.Sequential(ConcatPool(),
                         nn.BatchNorm1d(2 * nf), nn.Dropout(ps[0]), nn.Linear(2 * nf, hidden), nn.ReLU(inplace=True),
                         nn.BatchNorm1d(hidden), nn.Dropout(ps[1]), nn.Linear(hidden, nc))
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
    return head


class XResNet1d50(nn.Module):
    def __init__(self, num_classes, in_ch=12):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch)
        self.head = make_head(self.encoder.out_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
