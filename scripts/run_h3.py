"""Run the H3 experiment end to end (docs/h3_protocol_v1.md §5). Resumable: finished pieces are skipped.

  python scripts/run_h3.py --cache /tmp/cache --work /tmp/work --gpus 2 --archive /kaggle/working/work.tar

<work> must hold the finished core grid: ssl/SSL_ecg-mid_s{0..4}/ and runs/C1_b1_s{0..4}_lik0/ (with ckpt_best.pt).

Stages
  A  pretrain the seed-0 encoder of every policy-level (ecg/gen x low/mid/high); ecg-mid already exists
  B  linear probe of the six seed-0 encoders on fold 9, then select one level per policy (h3/selection.json);
     also the view-distortion diagnostic (h3/view_distortion.csv)
  C  pretrain seeds 1-4 of the two selected policy-levels
  D  fine-tune in the C1 configuration at 100%: the 3 x 2 grid at seed 0 and the selected levels at seeds 0-4.
     (ecg, mid) is the core C1 run and is reused, not rerun.
  E  held-out and seen corruption suite on every H3 run (and on the reused core runs if they lack it)
Outputs: ssl/, runs_h3/, h3/{probe_*.json, probe.csv, selection.json, view_distortion.csv, manifest.json}, logs/.
"""
import argparse, glob, hashlib, json, os, shlex, signal, subprocess, sys, tarfile, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from robust_ecg import h3  # noqa: E402


def tail(path, n=30):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return "(no log)"


def archive(work, path):
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as t:
        t.add(work, arcname=os.path.basename(os.path.normpath(work)))
    os.replace(tmp, path)


def run_jobs(jobs, gpus, logdir, label):
    """jobs: [(rid, argv, done_file)]. One job per GPU at a time. Stops at the first failure."""
    pending = [j for j in jobs if not os.path.exists(j[2])]
    print(f"stage {label}: {len(jobs)} jobs, {len(jobs) - len(pending)} finished, {len(pending)} to do", flush=True)
    running, failed, t_start = {}, None, time.time()

    def stop(sig, _):
        for _, p, _, _ in running.values():
            p.terminate()
        for _, p, _, _ in running.values():
            p.wait()
        print(f"stopped by signal {sig}; running jobs terminated (they resume from their checkpoints)", flush=True)
        sys.exit(130)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    while pending or running:
        for gpu in range(gpus):
            if gpu in running or failed or not pending:
                continue
            job = pending.pop(0)
            log = open(os.path.join(logdir, job[0] + ".log"), "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            running[gpu] = (job, subprocess.Popen([sys.executable, *job[1]], stdout=log, stderr=subprocess.STDOUT,
                                                  env=env), log, time.time())
        if not running:
            break
        time.sleep(2)
        for gpu, (job, p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close(); del running[gpu]
            lpath = os.path.join(logdir, job[0] + ".log")
            if p.returncode != 0 or not os.path.exists(job[2]):
                failed = job[0]
                print(f"FAILED {job[0]} (exit {p.returncode}). Last lines:\n{tail(lpath)}", flush=True)
            else:
                print(f"  done {job[0]} {(time.time() - t0) / 60:.1f} min (stage {(time.time() - t_start) / 60:.0f} min)"
                      f"  {tail(lpath, 1).strip()[:140]}", flush=True)
    if failed:
        sys.exit(f"stage {label} stopped: {failed} failed")


def rebuild(pattern, out_csv):
    rows = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--archive")
    ap.add_argument("--extra-ssl", default="", help="extra args for pretrain_simclr.py (smoke tests only)")
    ap.add_argument("--extra-ft", default="", help="extra args for train.py (smoke tests only)")
    ap.add_argument("--extra-eval", default="", help="extra args for eval_corruptions.py (smoke tests only)")
    a = ap.parse_args()
    W = a.work
    ssl, runs, runs_h3, hd, logs = (os.path.join(W, d) for d in ("ssl", "runs", "runs_h3", "h3", "logs"))
    for d in (runs_h3, hd, logs):
        os.makedirs(d, exist_ok=True)
    wk = ["--workers", str(a.workers)]
    enc = lambda p, l, s: os.path.join(ssl, h3.ssl_run_id(p, l, s), "encoder.pt")
    ssl_done = lambda p, l, s: os.path.join(ssl, h3.ssl_run_id(p, l, s), "done.json")

    def pretrain_job(p, l, s):
        return (h3.ssl_run_id(p, l, s), [os.path.join(HERE, "pretrain_simclr.py"), "--policy", p, "--strength", l,
                "--seed", str(s), "--cache", a.cache, "--runs", ssl, *wk, *shlex.split(a.extra_ssl)], ssl_done(p, l, s))

    def save_state():
        rebuild(os.path.join(ssl, "*", "done.json"), os.path.join(ssl, "registry_ssl.csv"))
        rebuild(os.path.join(runs_h3, "*", "done.json"), os.path.join(runs_h3, "registry.csv"))
        if a.archive:
            archive(W, a.archive)

    # core grid must be present: the (ecg, mid) arm is the core C1 runs
    for s in h3.SEEDS:
        for f in (ssl_done("ecg", "mid", s), os.path.join(runs, h3.core_run_id(s), "done.json"),
                  os.path.join(runs, h3.core_run_id(s), "ckpt_best.pt")):
            if not os.path.exists(f):
                sys.exit(f"core grid incomplete: {f} is missing. Restore the full work archive first.")

    # ---- A: seed-0 encoders -------------------------------------------------------------------------------
    run_jobs([pretrain_job(p, l, 0) for p in h3.POLICIES for l in h3.LEVELS], a.gpus, logs, "A (seed-0 pretraining)")
    save_state()

    # ---- B: linear probes, selection, distortion diagnostic --------------------------------------------------
    probe = lambda p, l: os.path.join(hd, f"probe_{p}-{l}.json")
    run_jobs([(f"probe_{p}-{l}", [os.path.join(HERE, "linear_probe.py"), "--cache", a.cache, "--encoder", enc(p, l, 0),
               "--out", probe(p, l), *wk], probe(p, l)) for p in h3.POLICIES for l in h3.LEVELS], a.gpus, logs,
             "B (linear probes)")
    scores = {p: {l: json.load(open(probe(p, l)))["val_macro_auroc"] for l in h3.LEVELS} for p in h3.POLICIES}
    selection = {p: h3.select_level(scores[p]) for p in h3.POLICIES}
    sel_file = os.path.join(hd, "selection.json")
    if os.path.exists(sel_file):
        old = json.load(open(sel_file))["selection"]
        if old != selection:
            sys.exit(f"selection changed between sessions ({old} -> {selection}); refusing to continue")
    else:
        json.dump(dict(selection=selection, scores=scores, tie_margin=h3.TIE_MARGIN, rule="highest fold-9 linear-probe "
                       "macro-AUROC; within the tie margin the lower level"), open(sel_file, "w"), indent=1)
    pd.DataFrame([dict(policy=p, level=l, val_macro_auroc=scores[p][l], selected=selection[p] == l)
                  for p in h3.POLICIES for l in h3.LEVELS]).to_csv(os.path.join(hd, "probe.csv"), index=False)
    print("probe scores:", json.dumps(scores), "\nselected:", selection, flush=True)

    dist_file = os.path.join(hd, "view_distortion.csv")
    if not os.path.exists(dist_file):
        import torch
        from robust_ecg import data
        X, meta = data.load_cache(a.cache, "ptbxl")
        mean, std = data.load_norm(a.cache)
        idx = np.flatnonzero(meta["strat_fold"].between(1, 8).to_numpy())
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pd.DataFrame([dict(policy=p, level=l, relative_distortion=h3.view_distortion(X, idx, mean, std, p, l, device=dev))
                      for p in h3.POLICIES for l in h3.LEVELS]).to_csv(dist_file, index=False)
    print(pd.read_csv(dist_file).round(4).to_string(index=False), flush=True)

    # ---- C: remaining seeds of the selected policy-levels -----------------------------------------------------
    run_jobs([pretrain_job(p, selection[p], s) for p in h3.POLICIES for s in h3.SEEDS if s > 0], a.gpus, logs,
             "C (pretraining, seeds 1-4 of the selected levels)")
    save_state()

    # ---- D: fine-tuning -------------------------------------------------------------------------------------
    need = h3.needed_runs(selection)
    jobs = []
    for p, l, s, _ in need:
        if h3.is_core(p, l):
            continue                                              # reused core run
        rid = h3.ft_run_id(p, l, s)
        jobs.append((rid, [os.path.join(HERE, "train.py"), "--arm", "C1", "--budget", "1.0", "--seed", str(s),
                           "--cache", a.cache, "--runs", runs_h3, "--pretrained", enc(p, l, s), "--tag", f"h3-{p}-{l}",
                           "--aug-strength", h3.FT_LEVEL, *wk, *shlex.split(a.extra_ft)],
                     os.path.join(runs_h3, rid, "done.json")))
    run_jobs(jobs, a.gpus, logs, "D (fine-tuning)")
    save_state()

    # ---- E: corruption suite ----------------------------------------------------------------------------------
    ev = [sys.executable, os.path.join(HERE, "eval_corruptions.py"), "--cache", a.cache, "--budgets", "1.0", "--arms", "C1",
          *wk, *shlex.split(a.extra_eval)]
    subprocess.run(ev + ["--runs", runs_h3], check=True)
    subprocess.run(ev + ["--runs", runs], check=True)             # no-op if notebook 04 already evaluated the core runs

    # ---- manifest ---------------------------------------------------------------------------------------------
    man = []
    for p, l, s, sel in need:
        d = h3.run_dir(W, p, l, s)
        done = json.load(open(os.path.join(d, "done.json")))
        cfg = json.load(open(os.path.join(d, "config.json")))
        sha = hashlib.sha256(open(enc(p, l, s), "rb").read()).hexdigest()
        if cfg.get("encoder_sha256") != sha or cfg.get("augment") != f"{h3.FT_POLICY}-{h3.FT_LEVEL}":
            sys.exit(f"{d}: encoder hash or fine-tuning augmentation does not match the protocol")
        if not os.path.exists(os.path.join(d, "corruptions.npz")):
            sys.exit(f"{d}: corruptions.npz missing")
        man.append(dict(policy=p, level=l, seed=s, in_primary=bool(sel), reused_core_run=h3.is_core(p, l),
                        run_dir=os.path.relpath(d, W).replace(os.sep, "/"), run_id=done["run_id"],
                        ssl_run=h3.ssl_run_id(p, l, s), encoder_sha256=sha))
    json.dump(dict(selection=selection, ft_augment=f"{h3.FT_POLICY}-{h3.FT_LEVEL}", runs=man),
              open(os.path.join(hd, "manifest.json"), "w"), indent=1)
    save_state()
    print(f"H3 complete: {len(man)} fine-tuned runs ({sum(m['in_primary'] for m in man)} in the primary contrast), "
          f"selection {selection}", flush=True)


if __name__ == "__main__":
    main()
