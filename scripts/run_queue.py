"""Run the core grid (RQ v2: S0, S1, C0, C1 x budgets x seeds, plus SSL pretraining) as a resumable queue.

  python scripts/run_queue.py --cache /tmp/cache --out /tmp/work --gpus 2 --archive /kaggle/working/work.tar

- One job per GPU at a time (CUDA_VISIBLE_DEVICES). A C0/C1 job starts only after its seed's SSL run has finished;
  meanwhile the free GPU takes the next supervised job.
- Finished runs (done.json) are skipped; interrupted runs resume from their own checkpoints (train.py and
  pretrain_simclr.py handle that).
- Each job's console output goes to <out>/logs/<run_id>.log. On the first failure no new jobs start, running jobs
  finish, and the tail of the failed log is printed; exit code 1.
- After every finished job, <out>/runs/registry.csv is rebuilt from the done.json files (concurrent jobs would
  otherwise race on appends) and, if --archive is given, <out> is re-archived to that tar (atomic replace).
"""
import argparse, glob, json, os, shlex, signal, subprocess, sys, tarfile, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def build_queue(seeds, budgets, cache, out, workers, one_pct=True, policy="ecg", strength="mid"):
    runs, ssl = os.path.join(out, "runs"), os.path.join(out, "ssl")
    q = []

    def ft(arm, b, seed):
        rid = f"{arm}_b{b:g}_s{seed}_lik0"
        args = ["train.py", "--arm", arm, "--budget", str(b), "--seed", str(seed), "--cache", cache,
                "--runs", runs, "--workers", str(workers)]
        dep = None
        if arm.startswith("C"):
            dep = f"SSL_{policy}-{strength}_s{seed}"
            args += ["--pretrained", os.path.join(ssl, dep, "encoder.pt")]
        return dict(rid=rid, dir=os.path.join(runs, rid), args=args, dep=dep)

    for seed in seeds:
        rid = f"SSL_{policy}-{strength}_s{seed}"
        q.append(dict(rid=rid, dir=os.path.join(ssl, rid), dep=None,
                      args=["pretrain_simclr.py", "--policy", policy, "--strength", strength, "--seed", str(seed),
                            "--cache", cache, "--runs", ssl, "--workers", str(workers)]))
        for arms in (("S1", "C1"), ("S0", "C0")):
            for b in budgets:
                q += [ft(arm, b, seed) for arm in arms]
    if one_pct:                                          # exploratory 1% budget (D10), last
        q += [ft(arm, 0.01, seed) for seed in seeds for arm in ("S1", "C1", "S0", "C0")]
    return q


def done(job, out):
    return os.path.exists(os.path.join(job["dir"], "done.json"))


def dep_ok(job, out):
    return job["dep"] is None or os.path.exists(os.path.join(out, "ssl", job["dep"], "done.json"))


def rebuild_registry(out):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "runs", "*", "done.json")))]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "runs", "registry.csv"), index=False)
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "ssl", "*", "done.json")))]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "ssl", "registry_ssl.csv"), index=False)


def archive(out, path):
    tmp = path + ".tmp"
    with tarfile.open(tmp, "w") as t:
        t.add(out, arcname=os.path.basename(os.path.normpath(out)))
    os.replace(tmp, path)


def tail(path, n=30):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return "(no log)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="directory holding runs/ and ssl/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--no-1pct", action="store_true")
    ap.add_argument("--gpus", type=int, default=1, help="parallel jobs, one per GPU id 0..N-1")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers per job")
    ap.add_argument("--archive", help="tar file refreshed after every finished job")
    ap.add_argument("--extra-ssl", default="", help="extra args for pretrain_simclr.py (smoke tests only)")
    ap.add_argument("--extra-ft", default="", help="extra args for train.py (smoke tests only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    q = build_queue(a.seeds, a.budgets, a.cache, a.out, a.workers, not a.no_1pct)
    pending = [j for j in q if not done(j, a.out)]
    print(f"{len(q)} jobs in the grid, {len(q) - len(pending)} finished, {len(pending)} to do, {a.gpus} parallel")
    if a.dry_run:
        for j in pending:
            print("  ", j["rid"], "(after " + j["dep"] + ")" if j["dep"] else "")
        return 0

    running, failed, n_done, t_start = {}, None, 0, time.time()

    def stop(sig, _):                      # never leave orphaned jobs writing into the run folders
        for _, p, _, _ in running.values():
            p.terminate()
        for _, p, _, _ in running.values():
            p.wait()
        print(f"stopped by signal {sig}; running jobs terminated (they resume from their checkpoints)", flush=True)
        sys.exit(130)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while pending or running:
        for gpu in range(a.gpus):
            if gpu in running or failed:
                continue
            job = next((j for j in pending if dep_ok(j, a.out)), None)
            if job is None:
                break
            pending.remove(job)
            log = open(os.path.join(a.out, "logs", job["rid"] + ".log"), "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            extra = shlex.split(a.extra_ssl if job["args"][0] == "pretrain_simclr.py" else a.extra_ft)
            p = subprocess.Popen([sys.executable, os.path.join(HERE, job["args"][0]), *job["args"][1:], *extra],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (job, p, log, time.time())
        if not running:
            if pending and not failed:     # only jobs whose SSL dependency is missing and not queued
                print("blocked:", [j["rid"] for j in pending]); return 1
            break
        time.sleep(5)
        for gpu, (job, p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close(); del running[gpu]
            lpath = os.path.join(a.out, "logs", job["rid"] + ".log")
            if p.returncode != 0 or not done(job, a.out):
                failed = job["rid"]
                print(f"FAILED {job['rid']} (exit {p.returncode}) on GPU {gpu}. Last lines:\n{tail(lpath)}", flush=True)
                continue
            n_done += 1
            rebuild_registry(a.out)
            if a.archive:
                archive(a.out, a.archive)
            last = tail(lpath, 1).strip()
            print(f"[{n_done} done, {len(pending)} left] GPU{gpu} {job['rid']} {(time.time() - t0) / 60:.1f} min "
                  f"(total {(time.time() - t_start) / 3600:.2f} h)  {last[:150]}", flush=True)
        if failed and not running:
            break
    rebuild_registry(a.out)
    if a.archive:
        archive(a.out, a.archive)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
