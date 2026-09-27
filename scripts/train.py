"""Supervised training for arms S0/S1 and fine-tuning for C0/C1 (research_question_v2.md §3; training_protocol_v1.md).

  python scripts/train.py --arm S0 --budget 1.0 --seed 0 \
      --cache /content/drive/MyDrive/robust-ecg-ssl/cache --runs /content/drive/MyDrive/robust-ecg-ssl/runs

Resumable: re-running the same command continues from the last checkpoint; a finished run is skipped.
Every finished run appends one row to <runs>/registry.csv and saves test predictions for bootstrapping.
"""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from robust_ecg import augment, data, labels, metrics, splits  # noqa: E402
from robust_ecg.models import XResNet1d50, count_params  # noqa: E402

ARMS = {"S0": dict(init="random", aug=None), "S1": dict(init="random", aug="ecg"),
        "C0": dict(init="ssl", aug=None), "C1": dict(init="ssl", aug="ecg")}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2**32)


def epoch_batches(n, bs, seed, epoch):
    perm = np.random.default_rng([seed, epoch]).permutation(n)
    return [perm[i:i + bs].tolist() for i in range(0, n - bs + 1, bs)] or [perm.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--pretrained", help="SSL encoder checkpoint (required for C0/C1)")
    ap.add_argument("--aug-strength", default="mid", choices=augment.STRENGTH)
    ap.add_argument("--min-likelihood", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--min-steps", type=int, default=2000)
    ap.add_argument("--n-evals", type=int, default=50, help="validation points per run")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="", help="suffix for non-protocol runs (e.g. smoke)")
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    if cfg["init"] == "ssl" and not a.pretrained:
        ap.error(f"{a.arm} needs --pretrained")

    run_id = f"{a.arm}_b{a.budget:g}_s{a.seed}_lik{a.min_likelihood:g}" + (f"_{a.tag}" if a.tag else "")
    rdir = os.path.join(a.runs, run_id); os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, "done.json")):
        print(f"{run_id}: already finished, skipping"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(a.seed)

    # ---- data ----
    X, meta = data.load_cache(a.cache, "ptbxl")
    scp = pd.read_csv(os.path.join(a.cache, "scp_statements.csv"), index_col=0)
    if list(scp.index) != json.load(open(os.path.join(a.cache, "ptbxl_statements.json"))):
        raise ValueError("scp_statements.csv does not match the statement list saved with the cache")
    Y, stmts = labels.ptbxl_multihot(meta, scp, a.min_likelihood)
    mean, std = data.load_norm(a.cache)
    tr_idx = splits.budget_indices(meta, a.budget, a.seed)
    va_idx = np.flatnonzero(meta["strat_fold"].to_numpy() == 9)
    te_idx = np.flatnonzero(meta["strat_fold"].to_numpy() == 10)
    aug = augment.Augment(cfg["aug"], a.aug_strength) if cfg["aug"] else None
    ds_tr = data.TrainCrops(X, Y, tr_idx, mean, std)          # augmentation is applied on the GPU per batch
    ev = lambda Xs, idx: DataLoader(data.EvalWindows(Xs, idx, mean, std), batch_size=64,
                                    num_workers=a.workers, pin_memory=dev.type == "cuda")

    steps_per_epoch = max(1, len(tr_idx) // a.bs)
    total = max(a.epochs * steps_per_epoch, a.min_steps)
    eval_every = max(1, total // a.n_evals)

    # ---- model / optimisation ----
    model = XResNet1d50(len(stmts)).to(dev)
    enc_sha = None
    if cfg["init"] == "ssl":
        enc_sha = hashlib.sha256(open(a.pretrained, "rb").read()).hexdigest()
        sd = torch.load(a.pretrained, map_location="cpu")
        model.encoder.load_state_dict(sd["encoder"] if "encoder" in sd else sd, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    step, best, log = 0, -1.0, []
    last = os.path.join(rdir, "ckpt_last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step, best, log = ck["step"], ck["best"], ck["log"]
        print(f"{run_id}: resumed at step {step}/{total}")
    else:
        json.dump(dict(vars(a), run_id=run_id, n_train=len(tr_idx), n_params=count_params(model),
                       total_steps=total, eval_every=eval_every, statements=len(stmts),
                       augment=aug.name if aug else None, encoder_sha256=enc_sha, device=str(dev),
                       torch=torch.__version__), open(os.path.join(rdir, "config.json"), "w"), indent=1)
    print(f"{run_id}: train={len(tr_idx)} records, {total} steps, eval every {eval_every}, device={dev}")

    # ---- train ----
    t0 = time.time()
    # one loader for all remaining steps: epoch e uses a fixed permutation seeded by (seed, e), so a
    # resumed run sees the same batch order; random crops/augmentations after a resume are re-drawn.
    epoch0, offset = divmod(step, steps_per_epoch)
    batches, e = [], epoch0
    while len(batches) < total - step + offset:
        batches += epoch_batches(len(tr_idx), a.bs, a.seed, e); e += 1
    batches = batches[offset:offset + total - step]
    g = torch.Generator(); g.manual_seed(a.seed * 100003 + step)
    dl = DataLoader(ds_tr, batch_sampler=batches, num_workers=a.workers, worker_init_fn=worker_init,
                    generator=g, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    model.train()
    if step < total:
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True)
            if aug is not None:
                xb = aug(xb)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = lossf(model(xb).float(), yb.to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % eval_every == 0 or step == total:
                P = data.predict_windows(model, ev(X, va_idx), dev)
                auc, n_lab = metrics.macro_auroc(Y[va_idx], P)
                log.append(dict(step=step, loss=loss.item(), val_macro_auroc=auc, minutes=(time.time() - t0) / 60))
                if auc > best:
                    best = auc; torch.save(model.state_dict(), os.path.join(rdir, "ckpt_best.pt"))
                torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                scaler=scaler.state_dict(), step=step, best=best, log=log), last + ".tmp")
                os.replace(last + ".tmp", last)
                print(f"step {step}/{total}  loss {float(loss):.4f}  val macro-AUROC {auc:.4f} "
                      f"(best {best:.4f}, {n_lab} labels)  {(time.time() - t0) / 60:.1f} min", flush=True)
                model.train()
            if step >= total:
                break

    # ---- evaluate best checkpoint ----
    model.load_state_dict(torch.load(os.path.join(rdir, "ckpt_best.pt"), map_location=dev))
    Pte = data.predict_windows(model, ev(X, te_idx), dev)
    res = dict(run_id=run_id, arm=a.arm, budget=a.budget, seed=a.seed, min_likelihood=a.min_likelihood,
               n_train=len(tr_idx), best_val_macro_auroc=best,
               best_step=max(log, key=lambda r: r["val_macro_auroc"])["step"], total_steps=total,
               flag_best_at_first_eval=max(log, key=lambda r: r["val_macro_auroc"])["step"] <= eval_every)
    res["test_macro_auroc"], res["test_n_labels"] = metrics.macro_auroc(Y[te_idx], Pte)
    Yte_e3, Pte_e3 = labels.e3_truth_ptbxl(Y[te_idx], stmts), labels.e3_scores(Pte, stmts)
    res["test_e3_macro_auroc"], _ = metrics.macro_auroc(Yte_e3, Pte_e3)
    save = dict(P_test=Pte.astype(np.float16), idx_test=te_idx)
    if os.path.exists(os.path.join(a.cache, "sph_X.npy")):
        Xs, sm = data.load_cache(a.cache, "sph")
        Ps = labels.e3_scores(data.predict_windows(model, ev(Xs, np.arange(len(sm))), dev), stmts)
        Ys = labels.sph_e3_labels(sm)
        res["sph_e3_macro_auroc"], _ = metrics.macro_auroc(Ys, Ps)
        res["e3_degradation"] = res["test_e3_macro_auroc"] - res["sph_e3_macro_auroc"]
        for k, v in zip(labels.E3_PRIMARY, metrics.per_label_auroc(Ys, Ps)):
            res[f"sph_auroc_{k}"] = v
        save["P_sph_e3"] = Ps.astype(np.float16)
    for k, v in zip(labels.E3_PRIMARY, metrics.per_label_auroc(Yte_e3, Pte_e3)):
        res[f"test_auroc_{k}"] = v
    res["train_minutes_last_session"] = log[-1]["minutes"]
    np.savez_compressed(os.path.join(rdir, "predictions.npz"), **save)
    pd.DataFrame(log).to_csv(os.path.join(rdir, "log.csv"), index=False)
    json.dump(res, open(os.path.join(rdir, "done.json"), "w"), indent=1, default=float)
    reg = os.path.join(a.runs, "registry.csv")
    pd.DataFrame([res]).to_csv(reg, mode="a", header=not os.path.exists(reg), index=False)
    os.remove(last)
    print(json.dumps({k: res[k] for k in ("run_id", "test_macro_auroc", "test_e3_macro_auroc")
                      + (("sph_e3_macro_auroc",) if "sph_e3_macro_auroc" in res else ())}, default=float))


if __name__ == "__main__":
    main()
