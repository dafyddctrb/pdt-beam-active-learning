"""Background worker: runs one job described by a run folder.

    python -m app.worker <run_dir>

Run-folder contract (read by the app):
  config.json        CaseStudyConfig
  measurements.json  {"displacement": {...}, "deterioration": {...}}  (single / al modes)
  job.json           {"mode": "single"|"repeated"|"al", "methods", "n_runs", "seed",
                      "reference", "strategies", "acq_fns", "conv_methods", "max_workers"}
  status.json        {"state": "running"|"done"|"failed"|"stopped", "step", "pid",
                      "started", "finished", "error"}
  run.log            worker stdout/stderr
"""
import json
import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.environ.get("PDT_APP_RUNS", os.path.join(REPO_ROOT, "app_runs"))


def _now():
    return datetime.now().isoformat(timespec="seconds")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_status(run_dir, **fields):
    path = os.path.join(run_dir, "status.json")
    status = read_json(path, {})
    status.update(fields)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def measurements_to_json(displacement, deterioration):
    conv = lambda d: {k: np.asarray(v).tolist() for k, v in d.items()}
    return {"displacement": conv(displacement), "deterioration": conv(deterioration)}


def measurements_from_json(d):
    conv = lambda x: {k: np.asarray(v, dtype=float) for k, v in x.items()}
    return conv(d["displacement"]), conv(d["deterioration"])


def create_run(cfg, job: dict, measurements=None, root=RUNS_ROOT) -> str:
    """Write a new run folder (config, job, measurements) and return its path."""
    run_dir = os.path.join(root, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job['mode']}")
    os.makedirs(run_dir)
    cfg.save(run_dir)
    with open(os.path.join(run_dir, "job.json"), "w") as f:
        json.dump(job, f, indent=2)
    if measurements is not None:
        with open(os.path.join(run_dir, "measurements.json"), "w") as f:
            json.dump(measurements_to_json(*measurements), f)
    write_status(run_dir, state="queued", step="", pid=None, started=_now(), finished=None, error=None)
    return run_dir


def start_run(run_dir) -> int:
    """Start the worker detached (own process group, survives page reloads)."""
    log = open(os.path.join(run_dir, "run.log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "app.worker", os.path.abspath(run_dir)], cwd=REPO_ROOT,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_status(run_dir, state="running", pid=p.pid)
    return p.pid


def worker_alive(run_dir) -> bool:
    """True if status.json's PID is alive AND is the worker for this run folder
    (guards against stale PIDs after a hard kill or reboot)."""
    pid = read_json(os.path.join(run_dir, "status.json"), {}).get("pid")
    if not pid:
        return False
    cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return "app.worker" in cmd and os.path.abspath(run_dir) in cmd


def mark_if_dead(run_dir, status: dict) -> dict:
    """A 'running' run whose worker is gone is marked failed (and returned updated)."""
    if status.get("state") == "running" and not worker_alive(run_dir):
        write_status(run_dir, state="failed", error="worker exited unexpectedly", finished=_now())
        status = read_json(os.path.join(run_dir, "status.json"), {})
    return status


def stop_run(run_dir):
    pid = read_json(os.path.join(run_dir, "status.json"), {}).get("pid")
    if pid and worker_alive(run_dir):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):   # already exited
            pass
    write_status(run_dir, state="stopped", finished=_now())


def main(run_dir):
    os.environ["PDT_NO_PLOTS"] = "1"
    sys.path.insert(0, REPO_ROOT)
    write_status(run_dir, state="running", pid=os.getpid(), step="starting")
    try:
        from case_study_config import CaseStudyConfig
        cfg = CaseStudyConfig.from_json(os.path.join(run_dir, "config.json"))
        job = read_json(os.path.join(run_dir, "job.json"))
        mode, seed = job.get("mode"), int(job.get("seed", 0))
        step = lambda text: (write_status(run_dir, step=text), print(f"=== {text} ===", flush=True))

        if mode == "single":
            from run_method_comparison import run_methods, set_seed
            meas = measurements_from_json(read_json(os.path.join(run_dir, "measurements.json")))
            set_seed(seed)
            run_methods(cfg, *meas, results_folder=run_dir, methods=job["methods"], on_step=step)
        elif mode == "repeated":
            from run_method_comparison import run_once, set_seed
            n = int(job["n_runs"])
            for i in range(n):
                step(f"run {i + 1}/{n}")
                set_seed(seed + i)
                run_once(cfg, run_dir)
        elif mode == "al":
            from run_active_learning_comparison import run_active_learning_comparison
            meas = measurements_from_json(read_json(os.path.join(run_dir, "measurements.json")))
            step("AL comparison")
            run_active_learning_comparison(cfg.beam_type, run_dir, case=cfg, measurements=meas,
                                           strategies=job.get("strategies"), acq_fns=job.get("acq_fns"),
                                           conv_methods=job.get("conv_methods"), max_workers=job.get("max_workers"))
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if job.get("reference") and mode != "al":
            step("reference posterior")
            from analyse_method_comparison import main as analyse
            analyse(results_folder=run_dir, beam_type=cfg.beam_type, reference=True,
                    ref_mcs_n=200_000, ref_grid_n=200)
        write_status(run_dir, state="done", step="done", finished=_now())
    except Exception as e:
        traceback.print_exc()
        write_status(run_dir, state="failed", error=f"{type(e).__name__}: {e}", finished=_now())
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
