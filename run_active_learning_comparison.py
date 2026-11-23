"""Comparison of active-learning settings for the surrogate of Section 4 (supporting study).

Runs one fixed data set of the cantilever investigation through every combination of updating
strategy (sequential, batch) x learning function x stopping criterion (the lists below), e.g. the
criterion of Eq. (19) (relative change of P_k, 'pf_stability') against min U >= 2. For each
combination it records the true-model calls, wall time and surrogate accuracy, and plots heatmaps,
learning curves and acquisition storyboards. The web app's "AL comparison" runs this module.

    python run_active_learning_comparison.py --results-folder al_comparison_results/my_run
"""
import copy
import json
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from utils.plot_style import plots_enabled
from case_study_config import CaseStudyConfig, add_config_cli_args, config_from_cli
from run_method_comparison import (
    run_sequential,
    run_batch,
)
from utils.plot_style import (set_academic_style, PRED_COLOR, TRUE_COLOR,
                              MEAS_COLOR, PRIOR_COLOR, TRUTH_COLOR)
from matplotlib.lines import Line2D

# ── Combinations to test ─────────────────────────────────────────────────────
# BUS-space acquisitions only; re-add the output-space baselines
# ('u_learning', 'two_sided_u', 'eff') if a head-to-head is needed again.
ACQ_FNS = ['bus_u_learning']   # first-order only
# Must match retrain_model's valid_conv exactly (behavior_models/gp_surrogate.py).
# 'seeded_delta' is the LSF-accuracy criterion: it stops on the re-anchored
# failure-fraction delta (seeded_delta_eps), not on relative Pf/beta change.
# Classic baselines first (min_u: Echard 2011 AK-MCS; beta_stability: Moustapha 2022
# benchmark), then the two new criteria (pf_stability, seeded_delta).
CONV_METHODS = ['min_u', 'pf_stability', 'pf_bound']#'beta_stability', 'pf_bound', 'beta_bound',] #, 'seeded_delta']
STRATEGIES = ['sequential', 'batch']
# Default relative-Pf tolerance; fallback for any method not in CONV_THRESHOLDS.
# Must live here (not a parent-side override) so spawned workers, which re-import
# this file, see it.
CONV_CRITERIA = 0.005
# Absolute failure-fraction threshold for convergence_method='seeded_delta'
# (forwarded as seeded_delta_eps; ignored by the relative Pf/beta methods).
# 0.1 (vs the original 0.05) — the eps sweep showed 0.05 over-enriches in the
# sequential strategy; 0.1 keeps local NRMSE < 0.3% at roughly half the points.
SEEDED_DELTA_EPS = 0.1

# Per-method convergence thresholds, each in its NATIVE input space:
#   min_u                                   -> U units (AK-MCS, Phi(-2) misclassification)
#   pf_stability / pf_bound / beta_stability / beta_bound / combined
#                                           -> RELATIVE-Pf tolerance. The beta_* tests
#       auto-convert this Pf tolerance to the equivalent beta tolerance at the live Pf
#       (beta_reltol_from_pf_reltol), so each beta_* method is matched to its pf_* twin
#       and the only difference is the space. Set beta_* independently here if you'd
#       rather not match (still interpreted as a Pf tolerance, then converted).
#   seeded_delta                            -> ABSOLUTE Pf-change (seeded_delta_eps)
CONV_THRESHOLDS = {
    'min_u':          2.0,
    'pf_stability':   0.01,
    'beta_stability': 0.01,
    'pf_bound':       0.1,
    'beta_bound':     0.1,
    'combined':       0.01,
    'seeded_delta':   SEEDED_DELTA_EPS,
}

# Number of parallel worker processes (each combo is independent). Threads pinned
# to 1 per worker (see _worker) to avoid BLAS oversubscription across processes.
MAX_WORKERS = 8

# Held-out surrogate-accuracy test set: a FIXED [E, I] cloud (shared by every combo,
# like the measurements) on which each trained surrogate is scored against the true
# deterministic deflection map disp = V*l^4/(8*E*I). N_HOLDOUT points; reproducible.
N_HOLDOUT = 500
HOLDOUT_SEED = 12345

# Known pathological combos to skip (optional)
SKIP_COMBOS = set()  # e.g. {('eff', 'min_u')}

# Shared RNG seed applied before EVERY combo so they all start from the identical
# initial seed design (and the same stochastic stream) — differences between configs
# then reflect the acquisition/convergence method, not a different starting point.
INIT_SEED = 0


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _metrics_path(results_folder: str) -> Path:
    return Path(results_folder) / "al_comparison" / "metrics.json"


def _load_metrics(results_folder: str) -> dict:
    p = _metrics_path(results_folder)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_metrics(results_folder: str, metrics: dict):
    p = _metrics_path(results_folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2)


def _combo_key(strategy: str, acq_fn: str, conv_method: str) -> str:
    return f"{strategy}__{acq_fn}__{conv_method}"


# ── Held-out surrogate-accuracy scoring ──────────────────────────────────────
def _build_holdout(cfg: dict):
    """Build the fixed held-out [E, I] set ONCE (in the parent), so every combo is
    scored on identical points. Spans the prior x deterioration domain the surrogate
    is queried on: E, lambda from the priors, a random measurement time per point
    (deterioration state -> I), then the true deflection the GP approximates."""
    np.random.seed(HOLDOUT_SEED)
    ref = cfg['case'].make_digital_twin()
    pmodel = ref.displacement_pmodel
    tf = np.asarray(cfg['timeframe'])
    E = np.asarray(ref.E_distribution.random(N_HOLDOUT)).reshape(-1)
    lam = np.asarray(ref.lambda_distribution.random(N_HOLDOUT)).reshape(-1)
    t_sel = tf[np.random.randint(0, len(tf), size=N_HOLDOUT)]      # span deterioration states
    DR = pmodel.get_loss_at_time(t=t_sel, lambda_dr=lam)
    I = np.asarray(pmodel.get_I_from_loss(loss=DR)).reshape(-1)
    X_test = np.column_stack((E, I))
    y_test = np.asarray(pmodel.get_beam_max_deflection(E=E, I=I)).reshape(-1)
    return X_test, y_test


def _score_surrogate(result_pdt, X_test, y_test):
    """Held-out RMSE + normalized RMSE (RMSE / mean|true|) of the final surrogate."""
    try:
        y_pred, _ = result_pdt.displacement_srg.predict(np.asarray(X_test, dtype=float), return_std=False)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_test, dtype=float).reshape(-1)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        denom = float(np.mean(np.abs(y_true))) or 1.0
        return rmse, rmse / denom
    except Exception as e:
        print(f"  [accuracy] scoring failed: {e}")
        return float('nan'), float('nan')


def _gain_per_point(history, key):
    """For accuracy column `key` ('nrmse' or 'nrmse_local'): returns (n_init, n_final,
    points_added, rel_acc_gain, acc_per_point, nrmse_final). rel_acc_gain is the
    fraction of the initial error removed; acc_per_point divides it by points added.
    init/final are the first/last snapshot with a finite value for that key."""
    pts = [h for h in (history or [])
           if h.get(key) is not None and not (isinstance(h[key], float) and np.isnan(h[key]))]
    if not pts:
        return None, None, None, None, None, None
    init, final = pts[0], pts[-1]
    added = int(final['n_train'] - init['n_train'])
    rel = acc = None
    if init[key] > 0:
        rel = round(float((init[key] - final[key]) / init[key]), 5)
        if added > 0:
            acc = round(float(rel / added), 6)
    return int(init['n_train']), int(final['n_train']), added, rel, acc, round(float(final[key]), 5)


def _campaign_efficiency(history):
    """Accuracy gain per training point over the campaign, for BOTH the global held-out
    set and the per-timestep local (current-posterior) set."""
    ni, nf, added, rel_g, acc_g, _ = _gain_per_point(history, 'nrmse')
    _, _, _, rel_l, acc_l, final_l = _gain_per_point(history, 'nrmse_local')
    return {'n_train_init': ni, 'n_train_final': nf, 'points_added': added,
            'rel_acc_gain': rel_g, 'acc_per_point': acc_g,
            'rel_acc_gain_local': rel_l, 'acc_per_point_local': acc_l,
            'nrmse_local': final_l}


# ── Run one combination ──────────────────────────────────────────────────────
def _scalar(x):
    return float(np.ravel(np.asarray(x, dtype=float))[0])


def _posterior_summary(pdt) -> dict:
    """Posterior per time step (mean, std, cumulative calls; last sub-step of each t)
    and the final-step samples of E, lambda and displacement, for scoring each
    combo against the reference posterior."""
    last = {int(t): subs[max(subs, key=int)] for t, subs in pdt.state_dict.items()}
    def thin(s, n=200):                              # a few samples per step, for the violins
        cols = {k: np.ravel(np.asarray(s[v].get("samples", []), dtype=float))
                for k, v in (("E", "E"), ("lambda", "lambda"), ("I", "I"), ("disp", "displacement"))}
        m = min(len(c) for c in cols.values())
        idx = np.linspace(0, m - 1, min(n, m)).astype(int) if m else []
        return {k: c[idx].tolist() for k, c in cols.items()}
    hist = {str(t): {**{v: {"mu": _scalar(s[v]["mu"]), "sigma": _scalar(s[v]["sigma"])}
                        for v in ("E", "lambda", "displacement")},
                     "calls": int(s.get("nr_likelihood_calls") or 0), "samples": thin(s)}
            for t, s in sorted(last.items())}
    s = last[max(last)]
    samples = {v: np.ravel(np.asarray(s[v].get("samples", []), dtype=float)).tolist()
               for v in ("E", "lambda", "displacement")}
    return {"posterior": hist, "final_samples": samples}


def _run_combo(PDT, PT, timeframe, results_folder, beam_type,
               displacement_measurement_dict, deterioration_rate_measurement_dict,
               samples_per_step, strategy, acq_fn, conv_method, X_test, y_test,
               base_al_kwargs: dict = None, p0: float = 0.1):
    """Run a single (strategy, acq_fn, conv_method) combo and return metrics dict.
    `base_al_kwargs` (from the case-study config) sets the shared AL settings; the
    combo's acquisition, convergence method and threshold override them."""
    # Pin the RNG so every combo draws the SAME initial seed design (set_parameters ->
    # initial_displacement_srg_model samples it) and shares the stochastic stream until
    # the method itself diverges. Without this each config starts from a different design.
    import torch
    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)

    pdt_copy = copy.deepcopy(PDT)
    pdt_copy._holdout = (X_test, y_test)   # enables per-timestep accuracy snapshots
    # Per-method threshold routed to the right kwarg in its native space.
    thr = CONV_THRESHOLDS.get(conv_method, CONV_CRITERIA)
    al_kwargs = dict(base_al_kwargs or {})
    al_kwargs.update(acquisition_function=acq_fn, convergence_method=conv_method)
    if conv_method == 'min_u':
        al_kwargs['min_u_threshold'] = thr
    elif conv_method == 'seeded_delta':
        al_kwargs['seeded_delta_eps'] = thr
    else:                                  # pf_*/beta_*/combined: Pf-relative (beta auto-converts)
        al_kwargs['convergence_criteria'] = thr

    t0 = time.time()
    try:
        if strategy == 'sequential':
            result_pdt = run_sequential(
                PDT=pdt_copy, PT=PT, timeframe=timeframe,
                results_folder=results_folder,
                model_type="srg", active_learning=True,
                beam_type=beam_type,
                displacement_measurement_dict=displacement_measurement_dict,
                deterioration_rate_measurement_dict=deterioration_rate_measurement_dict,
                samples_per_step=samples_per_step,
                al_kwargs=al_kwargs, p0=p0,
            )
        elif strategy == 'batch':
            result_pdt = run_batch(
                PDT_OG=pdt_copy, PT=PT, timeframe=timeframe,
                results_folder=results_folder,
                model_type="srg", active_learning=True,
                beam_type=beam_type,
                displacement_measurement_dict=displacement_measurement_dict,
                deterioration_rate_measurement_dict=deterioration_rate_measurement_dict,
                samples_per_step=samples_per_step,
                al_kwargs=al_kwargs, p0=p0,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        wall_time = time.time() - t0
        rmse, nrmse = _score_surrogate(result_pdt, X_test, y_test)
        history = list(getattr(result_pdt, 'accuracy_history', []) or [])
        m = {
            'strategy': strategy,
            'acquisition_function': acq_fn,
            'convergence_method': conv_method,
            'convergence_threshold': thr,
            'nr_likelihood_calls': int(result_pdt.nr_likelihood_calls),
            'rmse': round(rmse, 8),
            'nrmse': round(nrmse, 5),
            'wall_time_seconds': round(wall_time, 1),
            'accuracy_history': history,
            # Final surrogate training set (E, I), seed-first then AL-appended, plus
            # the seed-set size, so the acquisition storyboard can plot which points
            # each config selected over the shared held-out cloud + measurement LSF.
            'train_X': np.asarray(result_pdt.displacement_srg.X_train_current,
                                  dtype=float).tolist(),
            # Seed-set size: first n_init_train rows of train_X are the initial
            # design (set on the PDT in initial_displacement_srg_model); the rest
            # are AL-selected.
            'n_init_train': int(getattr(result_pdt, 'n_init_train', 0)),
            'error': None,
        }
        m.update(_campaign_efficiency(history))
        try:
            m.update(_posterior_summary(result_pdt))
        except Exception as e:                       # never lose a finished combo over this
            print(f"posterior summary skipped: {e}")
        return m
    except Exception as e:
        wall_time = time.time() - t0
        print(f"ERROR [{strategy}/{acq_fn} + {conv_method}]: {e}")
        traceback.print_exc()
        return {
            'strategy': strategy,
            'acquisition_function': acq_fn,
            'convergence_method': conv_method,
            'nr_likelihood_calls': -1,
            'rmse': float('nan'),
            'nrmse': float('nan'),
            'wall_time_seconds': round(wall_time, 1),
            'error': str(e),
        }


# ── Setup (shared across all combos) ─────────────────────────────────────────
def _draw_measurements(case, PT, disp_times, det_times):
    """Synthetic measurements from the physical twin (one data set for all combos)."""
    disp_mus, disp_sigmas = [], []
    for t_cur in disp_times:
        disp_mus.append(PT.measure_displacement(
            t_cur=np.array([t_cur]), noise_rel=case.displacement_noise_rel)[0][0])
        disp_sigmas.append(case.displacement_sigma)
    det_mus, det_sigmas = [], []
    for t_cur in det_times:
        mean, sigma = PT.measure_lambda(
            t_cur=t_cur, noise_rel=case.loss_noise_rel, sigma_lambda=case.sigma_lambda)
        det_mus.append(mean)
        det_sigmas.append(sigma)
    return ({"mean": np.array(disp_mus), "sigma": np.array(disp_sigmas), "time": np.array(disp_times)},
            {"mean": np.array(det_mus), "sigma": np.array(det_sigmas), "time": np.array(det_times)})


def build_setup(beam_type: str, case: CaseStudyConfig = None, measurements: tuple = None) -> dict:
    """Build the fixed experiment config once: beam params, timeframe, and the
    synthetic measurement dicts (drawn from the physical twin a single time so
    every combo sees identical measurements). Returns a picklable dict so worker
    processes can rebuild PDT/PT deterministically from params. Case-study
    settings come from `case` (see case_study_config.py)."""
    case = (case or CaseStudyConfig()).updated(beam_type=beam_type)
    timeframe = case.timeframe
    l, V, b, h, lambda_dr, E = case.l, case.V, case.b, case.h, case.true_lambda, case.true_E

    PT = case.make_physical_twin()

    samples_per_step = case.n_samples_srg
    measured_displacements_times = case.displacement_times
    measured_deterioration_rates_times = case.deterioration_times

    if measurements is not None:   # fixed data set given (e.g. by the web app)
        displacement_measurement_dict, deterioration_rate_measurement_dict = measurements
    else:
        displacement_measurement_dict, deterioration_rate_measurement_dict = _draw_measurements(
            case, PT, measured_displacements_times, measured_deterioration_rates_times)

    cfg = dict(beam_type=beam_type, timeframe=timeframe, l=l, V=V, b=b, h=h,
               E=E, lambda_dr=lambda_dr, samples_per_step=samples_per_step, case=case,
               displacement_measurement_dict=displacement_measurement_dict,
               deterioration_rate_measurement_dict=deterioration_rate_measurement_dict)
    cfg['X_test'], cfg['y_test'] = _build_holdout(cfg)   # fixed accuracy set, shared by all combos
    return cfg


def _make_pdt_pt(cfg: dict):
    """Rebuild PDT and PT deterministically from the config params."""
    PDT = cfg['case'].make_digital_twin()
    PT = cfg['case'].make_physical_twin()
    return PDT, PT


def _worker(task: tuple):
    """Top-level worker (picklable for spawn): pin threads, rebuild PDT/PT, run one combo."""
    import torch
    torch.set_num_threads(1)
    cfg, results_folder, beam_type, strategy, acq_fn, conv_method = task
    try:
        PDT, PT = _make_pdt_pt(cfg)
        m = _run_combo(PDT, PT, cfg['timeframe'], results_folder, beam_type,
                       cfg['displacement_measurement_dict'], cfg['deterioration_rate_measurement_dict'],
                       cfg['samples_per_step'], strategy, acq_fn, conv_method,
                       cfg['X_test'], cfg['y_test'],
                       base_al_kwargs=cfg['case'].al_kwargs, p0=cfg['case'].p0)
    except Exception as e:  # setup-level failure (combo-level already caught in _run_combo)
        m = {'strategy': strategy, 'acquisition_function': acq_fn, 'convergence_method': conv_method,
             'nr_likelihood_calls': -1, 'rmse': float('nan'), 'nrmse': float('nan'),
             'wall_time_seconds': 0.0, 'error': f'setup: {e}'}
    return _combo_key(strategy, acq_fn, conv_method), m


# ── Summary heatmap ──────────────────────────────────────────────────────────
def _fill_heatmap(ax, metrics, strategy, value_fn, fmt, title, cbar_label):
    """Fill one heatmap axis (ACQ_FNS x CONV_METHODS) for a single strategy."""
    data = np.full((len(ACQ_FNS), len(CONV_METHODS)), np.nan)
    annot = [['' for _ in CONV_METHODS] for _ in ACQ_FNS]
    for i, acq in enumerate(ACQ_FNS):
        for j, conv in enumerate(CONV_METHODS):
            m = metrics.get(_combo_key(strategy, acq, conv))
            if not m:
                continue
            if m.get('error') is not None:
                annot[i][j] = 'ERR'
                continue
            v = value_fn(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            data[i, j] = v
            annot[i][j] = fmt(v)
    sns.heatmap(data, ax=ax, annot=annot, fmt='', xticklabels=CONV_METHODS,
                yticklabels=ACQ_FNS, cmap='YlOrRd', linewidths=0.5, linecolor='white',
                cbar_kws={'label': cbar_label}, annot_kws={'size': 16})
    ax.set_xlabel('Convergence Method')
    ax.set_ylabel('')
    ax.set_title(title)


def plot_al_summary(metrics: dict, results_folder: str):
    """2x2 grid: cost (true model calls, top) vs surrogate accuracy (held-out
    NRMSE, bottom), sequential (left) vs batch (right). The cheap-AND-accurate
    methods land bright on top, dark on the bottom."""
    set_academic_style(presentation=True)

    # Short rows: the y-axis carries a single acquisition function, so most of the
    # vertical extent was dead space — keep each heatmap band thin and slide-wide.
    fig, axes = plt.subplots(3, 2, figsize=(15, 7.5), constrained_layout=True)

    for col, strategy in enumerate(STRATEGIES):
        _fill_heatmap(axes[0, col], metrics, strategy,
                      value_fn=lambda m: m.get('nr_likelihood_calls'),
                      fmt=lambda v: f'{int(v)}',
                      title=f'{strategy.capitalize()} — true model calls',
                      cbar_label='True model calls')
        _fill_heatmap(axes[1, col], metrics, strategy,
                      value_fn=lambda m: (m.get('nrmse') * 100.0 if m.get('nrmse') is not None else None),
                      fmt=lambda v: f'{v:.2f}%',
                      title=f'{strategy.capitalize()} — GLOBAL held-out NRMSE',
                      cbar_label='Global NRMSE (%)')
        _fill_heatmap(axes[2, col], metrics, strategy,
                      value_fn=lambda m: (m.get('nrmse_local') * 100.0 if m.get('nrmse_local') is not None else None),
                      fmt=lambda v: f'{v:.2f}%',
                      title=f'{strategy.capitalize()} — LOCAL (posterior) NRMSE',
                      cbar_label='Local NRMSE (%)')

    out_dir = Path(results_folder) / "plots" / "al_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(out_dir / f"al_comparison_summary_{time_id}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved cost-vs-accuracy heatmap to {out_dir}")


def plot_learning_curves(metrics: dict, results_folder: str):
    """Per-timestep accuracy trajectories vs cumulative training points, one line per
    (acq, conv). GLOBAL held-out NRMSE (top) and LOCAL current-posterior NRMSE (bottom),
    sequential (left) vs batch (right). Steeper / lower-left = more accuracy per training
    point. Colour = convergence method, linestyle = acquisition function."""
    set_academic_style(presentation=True)
    cmap = plt.get_cmap('tab10')
    conv_color = {c: cmap(i % 10) for i, c in enumerate(CONV_METHODS)}
    acq_style = {a: s for a, s in zip(ACQ_FNS, ['-', '--', ':', '-.'])}
    # (metric, efficiency key, row label, x-key, x-label, log-y). GLOBAL is plotted vs
    # cumulative training points (a cost axis); LOCAL is plotted vs measurement timestep
    # so the moving-posterior story is monotonic in x — cumulative points pile up at one
    # x once a method stops enriching, which drew as a misleading vertical spike. Log-y
    # on LOCAL keeps the 0.01–0.3% band visible instead of crushed against zero.
    rows = [('nrmse', 'acc_per_point', 'GLOBAL held-out',
             'n_train', 'Cumulative training points', False),
            ('nrmse_local', 'acc_per_point_local', 'LOCAL (posterior)',
             't', 'Measurement timestep', True)]

    # Wide-and-short for slides: compress the vertical extent, bump line/marker size.
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), constrained_layout=True)

    for r, (key, eff_key, label_row, xkey, xlabel, logy) in enumerate(rows):
        for c, strategy in enumerate(STRATEGIES):
            ax = axes[r, c]
            for acq in ACQ_FNS:
                for conv in CONV_METHODS:
                    m = metrics.get(_combo_key(strategy, acq, conv))
                    if not m or m.get('error') is not None:
                        continue
                    hist = [h for h in (m.get('accuracy_history') or [])
                            if h.get(key) is not None and not np.isnan(h[key])]
                    if len(hist) < 2:
                        continue
                    n = [h[xkey] for h in hist]
                    # Clip to a small positive floor so a perfect-fit (0%) point still
                    # renders on the log axis instead of being dropped.
                    y = [max(h[key] * 100.0, 1e-3) if logy else h[key] * 100.0 for h in hist]
                    app = m.get(eff_key)
                    label = f"{conv} · {acq.replace('bus_u_learning', 'u')}"
                    if app is not None:
                        label += f" ({app * 100:.2f}%/pt)"
                    ax.plot(n, y, marker='o', ms=4, lw=2.2, color=conv_color[conv],
                            linestyle=acq_style.get(acq, '-'), label=label)
            if logy:
                ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'{label_row} NRMSE (%)')
            ax.set_title(f'{strategy.capitalize()} updating — {label_row}')
            ax.legend(ncol=1, fontsize=11, handlelength=1.6, borderaxespad=0.3)

    out_dir = Path(results_folder) / "plots" / "al_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(out_dir / f"al_learning_curves_{time_id}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved learning-curve plot to {out_dir}")


# ── Acquisition storyboard (point selection per config) ──────────────────────
def plot_acquisition_storyboard(metrics: dict, cfg: dict, results_folder: str):
    """Problem-storyboard-style panels of WHICH points each config selected.

    Every panel shares the same backdrop — the FIXED held-out (E, I) cloud and the
    family of measurement limit-state contours E·I = V·l⁴/(8·y_obs) (one per
    displacement measurement, tracing the true deterioration path) — so the only
    thing that changes panel-to-panel is the config's own AL selection. The seed
    design is drawn faint; the AL-added points are highlighted and counted.

    One figure per strategy; rows = acquisition function, cols = convergence method.
    Combos missing point data (cached before this field existed, or errored) render
    as 'n/a' / 'ERR' over the shared backdrop.
    """
    set_academic_style()

    X_test = np.asarray(cfg['X_test'], dtype=float)
    E_all, I_all = X_test[:, 0], X_test[:, 1]
    l, V = float(cfg['l']), float(cfg['V'])
    meas_means = np.asarray(cfg['displacement_measurement_dict']['mean'], dtype=float)
    meas_means = meas_means[meas_means > 0]

    # True (E, I) trajectory: the ground-truth E is fixed; I deteriorates over the
    # campaign, so the true point at each measurement sits on that measurement's LSF
    # at E = E_true, i.e. I_true(t) = V·l⁴/(8·y_t·E_true). One gold star per measurement.
    E_true = float(cfg['E'])
    I_true_path = ((V * l ** 4) / (8.0 * meas_means * E_true)
                   if meas_means.size else np.array([]))

    # Shared axis limits: held-out cloud, expanded to contain every config's points
    # and the true trajectory.
    E_lo, E_hi = float(E_all.min()), float(E_all.max())
    I_lo, I_hi = float(I_all.min()), float(I_all.max())
    for m in metrics.values():
        tX = m.get('train_X')
        if not tX:
            continue
        tX = np.asarray(tX, dtype=float)
        E_lo, E_hi = min(E_lo, tX[:, 0].min()), max(E_hi, tX[:, 0].max())
        I_lo, I_hi = min(I_lo, tX[:, 1].min()), max(I_hi, tX[:, 1].max())
    E_lo, E_hi = min(E_lo, E_true), max(E_hi, E_true)
    if I_true_path.size:
        I_lo, I_hi = min(I_lo, float(I_true_path.min())), max(I_hi, float(I_true_path.max()))
    padE, padI = 0.05 * (E_hi - E_lo), 0.05 * (I_hi - I_lo)
    xlim = (E_lo - padE, E_hi + padE)
    ylim = (max(I_lo - padI, 1e-9 * max(I_hi, 1.0)), I_hi + padI)

    Egrid = np.linspace(xlim[0], xlim[1], 200)

    def _draw_backdrop(ax):
        ax.scatter(E_all, I_all, s=4, color=PRIOR_COLOR, alpha=0.45,
                   edgecolors='none', zorder=0)
        for y in meas_means:
            ax.plot(Egrid, (V * l ** 4) / (8.0 * y * Egrid),
                    color=MEAS_COLOR, lw=0.7, alpha=0.22, zorder=1)
        if I_true_path.size:
            E_col = np.full(I_true_path.shape, E_true)
            ax.plot(E_col, I_true_path, color=TRUE_COLOR, lw=1.0, alpha=0.6, zorder=5.5)
            ax.scatter(E_col, I_true_path, marker='*', s=70, c=TRUTH_COLOR,
                       edgecolors='black', linewidths=0.5, zorder=6)

    for strategy in STRATEGIES:
        fig, axes = plt.subplots(len(ACQ_FNS), len(CONV_METHODS),
                                 figsize=(3.1 * len(CONV_METHODS), 3.1 * len(ACQ_FNS)),
                                 sharex=True, sharey=True, squeeze=False,
                                 constrained_layout=True)
        for i, acq in enumerate(ACQ_FNS):
            for j, conv in enumerate(CONV_METHODS):
                ax = axes[i][j]
                _draw_backdrop(ax)
                m = metrics.get(_combo_key(strategy, acq, conv))
                title = f"{conv} · {acq.replace('bus_u_learning', 'u')}"
                if m and m.get('error') is None and m.get('train_X'):
                    tX = np.asarray(m['train_X'], dtype=float)
                    n0 = int(m.get('n_init_train') or 0)
                    seed, added = tX[:n0], tX[n0:]
                    ax.scatter(seed[:, 0], seed[:, 1], marker='^', s=20,
                               c=TRUE_COLOR, edgecolors='white', linewidths=0.3,
                               alpha=0.6, zorder=4)
                    ax.scatter(added[:, 0], added[:, 1], marker='o', s=32,
                               c=PRED_COLOR, edgecolors='black', linewidths=0.4,
                               zorder=5)
                    title += f"  (+{len(added)})"
                else:
                    ax.text(0.5, 0.5, 'ERR' if (m and m.get('error')) else 'n/a',
                            transform=ax.transAxes, ha='center', va='center',
                            color='0.55', fontsize=12)
                ax.set_title(title, fontsize='small')
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
                if i == len(ACQ_FNS) - 1:
                    ax.set_xlabel(r'$E$ [kN/m$^2$]')
                if j == 0:
                    ax.set_ylabel(r'$I$ [m$^4$]')

        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=PRIOR_COLOR,
                   markersize=7, label='held-out cloud (shared)'),
            Line2D([0], [0], color=MEAS_COLOR, lw=1.2, alpha=0.5,
                   label=r'measurement LSF  $E\,I = V l^4/(8\,y_{obs})$'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=TRUE_COLOR,
                   markersize=8, label='initial seed design'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=PRED_COLOR,
                   markeredgecolor='black', markersize=8,
                   label='AL-selected points (+N)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=TRUTH_COLOR,
                   markeredgecolor='black', markersize=12, label='true (E, I) path'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, -0.10))

        out_dir = Path(results_folder) / "plots" / "al_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(out_dir / f"al_acquisition_storyboard_{strategy}_{time_id}.png",
                    bbox_inches='tight')
        plt.close(fig)
        print(f"Saved acquisition storyboard ({strategy}) to {out_dir}")


# ── Main entry point ─────────────────────────────────────────────────────────
def run_active_learning_comparison(beam_type: str = "cantilever", results_folder: str = "al_comparison_results",
                      case: CaseStudyConfig = None, measurements: tuple = None,
                      strategies=None, acq_fns=None, conv_methods=None, max_workers=None):
    """Run every (strategy, acquisition, convergence) combo on one data set.
    `measurements` = (displacement_dict, deterioration_dict) to use a fixed data set;
    list arguments default to the module constants STRATEGIES / ACQ_FNS / CONV_METHODS."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    strategies, acq_fns = strategies or STRATEGIES, acq_fns or ACQ_FNS
    conv_methods, max_workers = conv_methods or CONV_METHODS, max_workers or MAX_WORKERS
    cfg = build_setup(beam_type, case, measurements)
    cfg['case'].save(results_folder)

    # ── Enumerate uncached combos ────────────────────────────────────────
    all_metrics = _load_metrics(results_folder)
    tasks = []
    for strategy in strategies:
        for acq_fn in acq_fns:
            for conv_method in conv_methods:
                if (acq_fn, conv_method) in SKIP_COMBOS:
                    continue
                key = _combo_key(strategy, acq_fn, conv_method)
                if key in all_metrics and all_metrics[key].get('error') is None:
                    print(f"[cached] {key}: {all_metrics[key]['nr_likelihood_calls']} calls")
                    continue
                tasks.append((cfg, results_folder, beam_type, strategy, acq_fn, conv_method))

    print(f"\nRunning {len(tasks)} combos on {max_workers} workers "
          f"({len(strategies)} strategies x {len(acq_fns)} acq x {len(conv_methods)} conv)\n")

    # ── Run in a process pool; save incrementally as each finishes ───────
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            key, m = fut.result()
            all_metrics[key] = m
            _save_metrics(results_folder, all_metrics)
            done += 1
            if m['error']:
                status = m['error']
            else:
                nr = m.get('nrmse')
                nr_s = f"{nr * 100:.2f}%" if isinstance(nr, (int, float)) and nr == nr else "n/a"
                status = f"{m['nr_likelihood_calls']} calls | NRMSE {nr_s} in {m['wall_time_seconds']}s"
            print(f"[{done}/{len(tasks)}] {key} -> {status}")

    # ── Summary plots + cost / accuracy / efficiency table ───────────────
    if plots_enabled():
        plot_al_summary(all_metrics, results_folder)
        plot_learning_curves(all_metrics, results_folder)
        plot_acquisition_storyboard(all_metrics, cfg, results_folder)
    def _pct(v):
        return f"{v * 100:.2f}%" if isinstance(v, (int, float)) and v is not None and v == v else "n/a"

    print(f"\n{'strategy':>10} {'acquisition':>22} {'convergence':>16} {'calls':>6} {'+pts':>5} "
          f"| {'gNRMSE':>7} {'g/pt':>7} | {'lNRMSE':>7} {'l/pt':>7}")
    for strategy in strategies:
        for acq in acq_fns:
            for conv in conv_methods:
                m = all_metrics.get(_combo_key(strategy, acq, conv))
                if not m:
                    continue
                if m.get('error'):
                    print(f"{strategy:>10} {acq:>22} {conv:>16} {'ERR':>6}")
                    continue
                pts = m.get('points_added')
                pts_s = str(pts) if pts is not None else "n/a"
                print(f"{strategy:>10} {acq:>22} {conv:>16} {m['nr_likelihood_calls']:>6} {pts_s:>5} "
                      f"| {_pct(m.get('nrmse')):>7} {_pct(m.get('acc_per_point')):>7} "
                      f"| {_pct(m.get('nrmse_local')):>7} {_pct(m.get('acc_per_point_local')):>7}")
    print("\n  (+pts = training points added; g* = GLOBAL held-out, l* = LOCAL posterior region;")
    print("   NRMSE = final error, /pt = % of initial error removed per training point)")
    print(f"\nAll done. Metrics: {_metrics_path(results_folder)}")


if __name__ == "__main__":
    import os
    import argparse
    p = argparse.ArgumentParser(description="Comparison of active-learning settings (learning function x stopping criterion)")
    p.add_argument("--results-folder", default=None)
    p.add_argument("--presentation", action="store_true",
                   help="Arial / larger-font presentation styling for all plots")
    add_config_cli_args(p)
    a = p.parse_args()
    case = config_from_cli(a)
    if a.presentation:
        os.environ["PLOT_PRESENTATION"] = "1"   # read by set_academic_style() in every plot
    results_folder = a.results_folder or (
        "al_comparison_results/run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_active_learning_comparison(beam_type=case.beam_type, results_folder=results_folder, case=case)
