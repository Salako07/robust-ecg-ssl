"""SimCLR pretraining of the xresnet1d50 encoder on PTB-XL folds 1-8 without labels (ssl_protocol_v1.md).

  python scripts/pretrain_simclr.py --policy ecg --strength mid --seed 0 \
      --cache /content/cache --runs /content/drive/MyDrive/robust-ecg-ssl/ssl

Resumable (checkpoint every --ckpt-every epochs). The encoder from the FINAL epoch is saved as encoder.pt;
no labels are used for selection. Finished runs append a row to <runs>/registry_ssl.csv.
"""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg import augment, data  # noqa: E402
from robust_ecg.models import SimCLRNet, count_params, nt_xent  # noqa: E402


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=augment.POLICIES)
    ap.add_argument("--strength", default="mid", choices=augment.STRENGTH)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=0.5, help="probability of each transform")
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="timing only: stop after N (>3) steps, save nothing")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.max_steps and a.max_steps <= 3:
        ap.error("--max-steps must be > 3")

    run_id = f"SSL_{a.policy}-{a.strength}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    rdir = os.path.join(a.runs, run_id)
    if not a.max_steps:
        os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, "done.json")):
        print(f"{run_id}: already finished, skipping"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    X, meta = data.load_cache(a.cache, "ptbxl")
    idx = np.flatnonzero(meta["strat_fold"].between(1, 8).to_numpy())      # folds 1-8 only, labels unused
    mean, std = data.load_norm(a.cache)
    ds = data.PairCrops(X, idx, mean, std)
    aug = augment.Augment(a.policy, a.strength, p=a.p)
    steps_per_epoch = len(idx) // a.bs
    total = a.epochs * steps_per_epoch
    warm = a.warmup_epochs * steps_per_epoch

    net = SimCLRNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    epoch, log = 0, []
    last = os.path.join(rdir, "ckpt_last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        epoch, log = ck["epoch"], ck["log"]
        torch.set_rng_state(ck["rng"]["torch"]); np.random.set_state(ck["rng"]["numpy"]); random.setstate(ck["rng"]["py"])
        if dev.type == "cuda" and ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"{run_id}: resumed at epoch {epoch}/{a.epochs}")
    elif not a.max_steps:
        json.dump(dict(vars(a), run_id=run_id, n_records=len(idx), n_params=count_params(net),
                       steps_per_epoch=steps_per_epoch, total_steps=total, augment=aug.name, device=str(dev),
                       torch=torch.__version__), open(os.path.join(rdir, "config.json"), "w"), indent=1)
    print(f"{run_id}: {len(idx)} unlabelled records, {steps_per_epoch} steps/epoch, {a.epochs} epochs, device={dev}")

    t0, step_count = time.time(), 0
    while epoch < a.epochs:
        g = torch.Generator(); g.manual_seed(a.seed * 100003 + epoch)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True, generator=g, num_workers=a.workers,
                        worker_init_fn=worker_init, pin_memory=dev.type == "cuda")
        net.train(); tl, ta, n = 0.0, 0.0, 0
        for v1, v2 in dl:
            v1 = aug(v1.to(dev, non_blocking=True)); v2 = aug(v2.to(dev, non_blocking=True))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                z1, z2 = net(v1), net(v2)
            loss, acc = nt_xent(z1.float(), z2.float(), a.tau)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); ta += acc.item(); n += 1; step_count += 1
            if step_count == 3:
                t_warm = time.time()      # timing excludes the first steps (cuDNN autotune, worker start-up)
            if a.max_steps and step_count >= a.max_steps:
                sec = (time.time() - t_warm) / max(1, step_count - 3)
                print(f"timing: {sec:.3f} s/step -> {sec * total / 3600:.2f} h for {a.epochs} epochs "
                      f"({total} steps); last loss {float(loss):.3f}, top-1 {float(acc):.3f}")
                return
        epoch += 1
        log.append(dict(epoch=epoch, loss=tl / n, top1=ta / n, lr=sched.get_last_lr()[0],
                        minutes=(time.time() - t0) / 60))
        if epoch % a.ckpt_every == 0 or epoch == a.epochs:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, log=log,
                            rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), py=random.getstate(),
                                     cuda=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)),
                       last + ".tmp")
            os.replace(last + ".tmp", last)
        print(f"epoch {epoch}/{a.epochs}  loss {tl / n:.4f}  top-1 {ta / n:.3f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)

    enc = os.path.join(rdir, "encoder.pt")
    torch.save({"encoder": net.encoder.state_dict()}, enc)
    pd.DataFrame(log).to_csv(os.path.join(rdir, "log.csv"), index=False)
    res = dict(run_id=run_id, policy=a.policy, strength=a.strength, seed=a.seed, epochs=a.epochs, bs=a.bs,
               tau=a.tau, final_loss=log[-1]["loss"], final_top1=log[-1]["top1"],
               minutes_last_session=log[-1]["minutes"],
               encoder_sha256=hashlib.sha256(open(enc, "rb").read()).hexdigest())
    json.dump(res, open(os.path.join(rdir, "done.json"), "w"), indent=1)
    reg = os.path.join(a.runs, "registry_ssl.csv")
    pd.DataFrame([res]).to_csv(reg, mode="a", header=not os.path.exists(reg), index=False)
    os.remove(last)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
