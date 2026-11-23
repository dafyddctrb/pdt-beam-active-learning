"""Read run folders written by app/worker.py into DataFrames for plotting."""
import glob
import os

import numpy as np
import pandas as pd

from utils.model_calls import cumulative_calls
from app.worker import REPO_ROOT, RUNS_ROOT, mark_if_dead, read_json

STUDIES_ROOT = os.environ.get("PDT_STUDIES", os.path.join(REPO_ROOT, "results"))

METHOD_LABELS = {
    "CD_Sequential Updating with Physics Based Model": "Seq PM",
    "CD_Sequential Updating with SRG Prediction and Active Learning": "Seq SRG",
    "CD_Full Updating with Physics Based Model": "Batch PM",
    "CD_Full Updating with SRG Prediction and Active Learning": "Batch SRG",
}
METHOD_ORDER = ["Seq PM", "Seq SRG", "Batch PM", "Batch SRG"]

# Names follow Moustapha, Marelli & Sudret (2022), "Active learning for structural
# reliability: survey and benchmark", and Echard et al. (2011) for the U-function.
ACQ_NAMES = {
    "bus_u_learning": "U-function, first order",
    "bus_u_learning_2nd": "U-function, second order",
}
CONV_NAMES = {
    "min_u": "Learning-function based (min U ≥ 2)",
    "pf_stability": "Stability of Pf",
    "pf_bound": "Bounds on Pf",
    "beta_stability": "Stability of β",
    "beta_bound": "Bounds on β",
    "combined": "Combined: stability and bounds on β",
    "seeded_delta": "Re-anchored change of the failure fraction",
}
# short forms for narrow places (column titles)
CONV_SHORT = {"min_u": "min U ≥ 2", "pf_stability": "Stability of Pf", "pf_bound": "Bounds on Pf",
              "beta_stability": "Stability of β", "beta_bound": "Bounds on β", "combined": "Combined (β)",
              "seeded_delta": "Re-anchored ε"}
STRATEGY_NAMES = {"sequential": "Sequential", "batch": "Batch"}


def list_runs(root=RUNS_ROOT, studies_root=STUDIES_ROOT):
    """App runs (newest first), then the studies run from the command line (run_method_comparison.py,
    e.g. results/sigma_d_0p02): a study with one run opens as a single data set,
    otherwise as a repeated study. Runs made with the old (compounded) deterioration law are
    left out: their truth and reference would be drawn with the current law."""
    runs = []
    from case_study_config import is_legacy_config
    legacy = lambda d: is_legacy_config(read_json(os.path.join(d, "config.json"), {}))
    for d in sorted(glob.glob(os.path.join(root, "*")), reverse=True):
        job = read_json(os.path.join(d, "job.json"))
        status = mark_if_dead(d, read_json(os.path.join(d, "status.json"), {}))
        if job and not (legacy(d) and status.get("state") not in ("running", "queued")):
            runs.append({"dir": d, "name": os.path.basename(d), "mode": job.get("mode"),
                         "state": status.get("state"), "started": status.get("started")})
    for d in sorted(glob.glob(os.path.join(studies_root, "*"))):
        if not os.path.exists(os.path.join(d, "config.json")) or legacy(d):
            continue
        files = _state_files(d)
        n = max((len(f) for f in files.values()), default=0)
        if n:
            runs.append({"dir": d, "name": "results/" + os.path.basename(d), "study": True, "n_reps": n,
                         "mode": "single" if n == 1 else "repeated", "state": "", "started": None})
    return runs


def _state_files(run_dir):
    """{method label: [state-dict json files, one per run, oldest first]}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*", "*", "*", "state_dict_run_*.json"))):
        label = METHOD_LABELS.get(os.path.basename(os.path.dirname(f)))
        if label:
            out.setdefault(label, []).append(f)
    return out


def _last_substeps(path):
    """{t: state at the last sub-step of t}, or None if the file is unreadable."""
    d = read_json(path)
    if not d:
        return None
    return {int(t): subs[max(subs, key=int)] for t, subs in d.items()}


def load_states(run_dir) -> pd.DataFrame:
    rows = []
    for method, files in _state_files(run_dir).items():
        for rep, f in enumerate(files):
            for t, s in (_last_substeps(f) or {}).items():
                rows.append({"method": method, "rep": rep, "t": t,
                             "E_mu": s["E"]["mu"], "E_sigma": s["E"]["sigma"],
                             "lambda_mu": s["lambda"]["mu"], "lambda_sigma": s["lambda"]["sigma"],
                             "disp_mu": s["displacement"]["mu"], "disp_sigma": s["displacement"]["sigma"],
                             "calls": s.get("nr_likelihood_calls")})
    return pd.DataFrame(rows).sort_values(["method", "rep", "t"]) if rows else pd.DataFrame()


def add_cumulative_calls(states: pd.DataFrame, cfg) -> pd.DataFrame:
    """Adds `calls_cum`, the true-model calls spent up to and including t
    (see utils.model_calls.cumulative_calls)."""
    out = []
    for (method, rep), g in states.sort_values("t").groupby(["method", "rep"], sort=False):
        out.append(g.assign(calls_cum=cumulative_calls(method, g["t"].to_numpy(), g["calls"].to_numpy(),
                                                       cfg.deterioration_times)))
    return pd.concat(out).sort_values(["method", "rep", "t"]) if out else states


def load_samples(run_dir, method, rep, t) -> dict:
    files = _state_files(run_dir).get(method, [])
    s = (_last_substeps(files[rep]) or {}).get(int(t)) if rep < len(files) else None
    if s is None:
        return {}
    train = s.get("surrogate_train_X")
    return {"E": np.asarray(s["E"].get("samples", []), dtype=float),
            "lambda": np.asarray(s["lambda"].get("samples", []), dtype=float),
            "train_X": np.asarray(train, dtype=float) if train else None,
            "n_init": s.get("surrogate_n_init_train", 0)}


def true_curves(cfg) -> pd.DataFrame:
    """True displacement and second moment of area I of the physical twin at t = 0 .. t_end."""
    from reference_posterior import make_beam, displacement
    beam = make_beam(cfg.beam_type, **cfg.beam_kwargs)
    t = np.concatenate([[0.0], np.asarray(cfg.timeframe, dtype=float)])
    lam = np.array([cfg.true_lambda])
    disp = [float(np.ravel(displacement(beam, np.array([cfg.true_E]), lam, tt))[0]) for tt in t]
    I = [float(np.ravel(beam.get_I_at_time(lambda_dr=lam, t_cur=tt))[0]) for tt in t]
    return pd.DataFrame({"t": t, "disp": disp, "I": I})


def load_sample_history(run_dir, method, rep=0) -> pd.DataFrame:
    """Posterior samples at every time step (last sub-step) of one method and run,
    long format: t, E, lambda, I, disp."""
    files = _state_files(run_dir).get(method, [])
    steps = _last_substeps(files[rep]) if rep < len(files) else None
    rows = []
    for t, s in sorted((steps or {}).items()):
        cols = {k: np.ravel(np.asarray(s[v].get("samples", []), dtype=float))
                for k, v in (("E", "E"), ("lambda", "lambda"), ("I", "I"), ("disp", "displacement"))}
        n = min(len(c) for c in cols.values())
        if n:
            rows.append(pd.DataFrame({"t": t, **{k: c[:n] for k, c in cols.items()}}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_measurements(run_dir, rep=0):
    """The run's measurements; for command-line studies (no measurements.json) the data of
    run `rep` (0-based), read back from its state file (the i-th file of every method belongs to
    the same data set)."""
    m = read_json(os.path.join(run_dir, "measurements.json"))
    if m or not _state_files(run_dir):
        return m
    from reference_posterior import extract_measurements
    files = next(iter(_state_files(run_dir).values()))
    if rep >= len(files):
        return None
    data = read_json(files[rep])
    if not data:
        return None
    x = extract_measurements(data)
    from case_study_config import CaseStudyConfig        # the readings follow the configured schedule
    cfg = CaseStudyConfig.from_json(os.path.join(run_dir, "config.json"))
    det_t = [float(t) for t in cfg.deterioration_times][:len(x["prop_mean"])]
    return {"displacement": {"time": x["disp_time"].tolist(), "mean": x["disp_mean"].tolist(),
                             "sigma": x["disp_sigma"].tolist()},
            "deterioration": {"time": det_t,
                              "mean": np.asarray(x["prop_mean"]).tolist(),
                              "sigma": np.asarray(x["prop_sigma"]).tolist()}}


def _meas_up_to(cfg, meas: dict, t: float) -> dict:
    """Measurement dict for reference_posterior, keeping the data with time <= t."""
    d, q = meas["displacement"], meas["deterioration"]
    dk = [i for i, tt in enumerate(d["time"]) if tt <= t]
    qk = [i for i, tt in enumerate(q["time"]) if tt <= t]
    return {"E_prior": (cfg.true_E * cfg.prior_E_factor, cfg.true_E * cfg.prior_E_factor * cfg.prior_E_cov),
            "lambda_prior": (cfg.true_lambda * cfg.prior_lambda_factor,
                             cfg.true_lambda * cfg.prior_lambda_factor * cfg.prior_lambda_cov),
            "disp_time": np.array([d["time"][i] for i in dk], dtype=float),
            "disp_mean": np.array([d["mean"][i] for i in dk], dtype=float),
            "disp_sigma": np.array([d["sigma"][i] for i in dk], dtype=float),
            "prop_mean": np.array([q["mean"][i] for i in qk], dtype=float),
            "prop_sigma": np.array([q["sigma"][i] for i in qk], dtype=float)}


def reference_at(cfg, meas: dict, t: float, n: int = 150, n_resample: int = 10) -> dict:
    """Numerical reference posterior (grid integration) given the data measured up to t."""
    from reference_posterior import reference_grid
    return reference_grid(_meas_up_to(cfg, meas, t), beam_type=cfg.beam_type, n=n, n_resample=n_resample,
                          t_pred=max(t, 1e-9), beam_kwargs=cfg.beam_kwargs)


def reference_samples(cfg, meas: dict, n: int = 120, n_per_t: int = 300) -> pd.DataFrame:
    """Draws from the numerical reference posterior at t = 0 (the prior) and at every time step, given the
    data measured up to t; long format: t, E, lambda, I, disp."""
    from reference_posterior import make_beam
    beam = make_beam(cfg.beam_type, **cfg.beam_kwargs)
    rows = []
    for t in [0] + list(cfg.timeframe):
        r = reference_at(cfg, meas, t, n=n, n_resample=n_per_t)
        lam = np.ravel(r["lambda_samples"])
        rows.append(pd.DataFrame({"t": t, "E": np.ravel(r["E_samples"]), "lambda": lam,
                                  "I": np.ravel(beam.get_I_at_time(lambda_dr=lam, t_cur=t)),
                                  "disp": np.ravel(r["displacement_samples"])}))
    return pd.concat(rows, ignore_index=True)


def al_sample_history(metrics: pd.DataFrame) -> dict:
    """{combo: samples per time step (t, E, lambda, I, disp)} for AL runs that stored them
    (runs from before this feature only stored mean and std)."""
    out = {}
    for r in metrics.to_dict("records"):
        post = r.get("posterior")
        rows = [pd.DataFrame({"t": int(t), **{k: s["samples"][k] for k in ("E", "lambda", "I", "disp")}})
                for t, s in (post.items() if isinstance(post, dict) else []) if "samples" in s]
        if rows:
            out[r["combo"]] = pd.concat(rows, ignore_index=True)
    return out


def reference_I_cloud(cfg, meas: dict, n_per_t: int = 150) -> pd.DataFrame:
    """Reference-posterior draws mapped to the surrogate's input space (E, I): at each
    time step, (E, lambda) drawn from the posterior given the data up to t, with
    I = I(lambda, t). This is the region the surrogate has to be accurate in."""
    from reference_posterior import make_beam
    beam = make_beam(cfg.beam_type, **cfg.beam_kwargs)
    rows = []
    for t in cfg.timeframe:
        r = reference_at(cfg, meas, t, n=120, n_resample=n_per_t)
        I = np.ravel(beam.get_I_at_time(lambda_dr=r["lambda_samples"], t_cur=t))
        rows.append(pd.DataFrame({"t": t, "E": r["E_samples"], "I": I}))
    return pd.concat(rows, ignore_index=True)


def al_posteriors(metrics: pd.DataFrame) -> pd.DataFrame:
    """Posterior history of every AL combination (stored in metrics.json by
    run_active_learning_comparison): combo, strategy, t, E_mu, ..., calls."""
    rows = []
    for r in metrics.to_dict("records"):
        post = r.get("posterior")
        for t, s in (post.items() if isinstance(post, dict) else []):
            rows.append({"combo": r["combo"], "strategy": r["strategy"], "t": int(t),
                         "E_mu": s["E"]["mu"], "E_sigma": s["E"]["sigma"],
                         "lambda_mu": s["lambda"]["mu"], "lambda_sigma": s["lambda"]["sigma"],
                         "disp_mu": s["displacement"]["mu"], "disp_sigma": s["displacement"]["sigma"],
                         "calls": s["calls"]})
    return pd.DataFrame(rows).sort_values(["combo", "t"]) if rows else pd.DataFrame()


def load_al_metrics(run_dir) -> pd.DataFrame:
    m = read_json(os.path.join(run_dir, "al_comparison", "metrics.json"), {})
    df = pd.DataFrame(list(m.values()))
    if not df.empty:
        name = lambda d, k: d.get(k, k)
        df["combo"] = [f"{name(STRATEGY_NAMES, s)}, {name(CONV_NAMES, c)}"
                       + (f", {name(ACQ_NAMES, a)}" if df.acquisition_function.nunique() > 1 else "")
                       for s, a, c in zip(df.strategy, df.acquisition_function, df.convergence_method)]
    return df


def load_csv(run_dir, name):
    p = os.path.join(run_dir, "plots", name)
    return pd.read_csv(p, header=[0, 1], index_col=0) if name.endswith("summary.csv") and os.path.exists(p) \
        else (pd.read_csv(p) if os.path.exists(p) else None)


def study_data(run_dir, cfg, states: pd.DataFrame, n_plot: int = 5000):
    """Final-time-step summary of a repeated study, as in the paper (Table 2, Figure 7).

    Returns (table, samples, calls):
      table   - posterior mean and std of E, lambda and the displacement, pooled over all samples of
                all runs, for the reference (one grid-integration posterior per data set) and each
                method, with the cumulative true-model calls (mean and std over runs);
      samples - long format (who, kind = prior | posterior, var = E | lambda | disp, value), at most
                `n_plot` pooled draws each, for violins;
      calls   - cumulative true-model calls per method and run.
    """
    os.environ.setdefault("PDT_NO_PLOTS", "1")
    from analyse_method_comparison import find_json_files, parse_results
    from reference_posterior import reference_grid
    df = pd.DataFrame([r for r in (parse_results(f, cfg.deterioration_times) for f in find_json_files(run_dir))
                       if r is not None])
    post_cols = {"E": "E_samples_posterior", "lambda": "lambda_samples_posterior",
                 "disp": "Displacement_samples_posterior"}
    prior_cols = {"E": "E_samples_prior", "lambda": "lambda_samples_prior", "disp": "Displacement_samples_prior"}
    cat = lambda arrays: np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    rng = np.random.default_rng(0)
    thin = lambda v: v if len(v) <= n_plot else rng.choice(v, n_plot, replace=False)
    refs = [reference_grid(m, beam_type=cfg.beam_type, n=150, n_resample=2000, t_pred=cfg.t_end,
                           beam_kwargs=cfg.beam_kwargs)
            for m in df.groupby("measurement_key", sort=False)["measurements"].first()]
    pooled = {"Reference": {"E": cat([r["E_samples"] for r in refs]), "lambda": cat([r["lambda_samples"] for r in refs]),
                            "disp": cat([r["displacement_samples"] for r in refs])}}
    priors = {}
    for method in METHOD_ORDER:
        g = df[df.Method == method]
        if len(g):
            pooled[method] = {k: cat(g[c]) for k, c in post_cols.items()}
            priors[method] = {k: cat(g[c]) for k, c in prior_cols.items()}
    # The prior is the same for every method. Take it from a physics-model method: for the surrogate
    # methods the stored prior displacement is the untrained surrogate's prediction, not the prior.
    pm = next((m for m in ("Seq PM", "Batch PM") if m in priors), next(iter(priors)))
    priors = {who: priors[pm] for who in ["Reference", *priors]}

    fin = add_cumulative_calls(states, cfg)
    fin = fin.loc[fin.groupby(["method", "rep"]).t.idxmax()]
    calls = fin[["method", "rep", "calls_cum"]].reset_index(drop=True)
    rows = {}
    for who, d in pooled.items():
        rows[who] = {f"{k}_{s}": f(v) for k, v in d.items() for s, f in (("mu", np.mean), ("sigma", np.std))}
        c = calls[calls.method == who].calls_cum
        if len(c):
            rows[who].update(calls_mu=c.mean(), calls_sigma=c.std())
    table = pd.DataFrame(rows).T
    table.attrs["n_data_sets"] = len(refs)
    samples = pd.concat([pd.DataFrame({"who": who, "kind": kind, "var": k, "value": thin(v)})
                         for kind, src in (("prior", priors), ("posterior", pooled))
                         for who, d in src.items() for k, v in d.items()], ignore_index=True)
    return table, samples, calls
