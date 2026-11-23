"""Analysis of a results folder written by run_method_comparison.py (paper, Section 5.2.3).

Reads every run's state files, computes the exact reference posterior of each data set
(reference_posterior.py: grid quadrature, cross-checked by Monte Carlo) and writes to
<results-folder>/plots/:
  comparison_plot.png          final posteriors of E, lambda and the displacement over all runs,
                               with the true-model calls (the paper's figure of the final posteriors)
  reference_metrics.csv        accuracy of every method against the reference (mean error, std
                               ratio, Wasserstein-1, energy distance), the basis of Table 2
  reference_metrics_summary.csv, reference_metrics.png, reference_accuracy_vs_cost.png

    python analyse_method_comparison.py --results-folder results/sigma_d_0p02
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from case_study_config import CaseStudyConfig
from utils.model_calls import cumulative_calls
from reference_posterior import (extract_measurements, measurement_key, reference_mcs,
                                 reference_grid, compare_to_reference, pooled_marginal_density)
from utils.plot_style import (set_academic_style, legend_below, PRED_COLOR,
                              TRUE_COLOR, PRIOR_COLOR, MEAS_COLOR, TRUTH_COLOR)

def find_json_files(results_dir):
    json_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.startswith("state_dict_run_") and file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def get_true_values(cfg: CaseStudyConfig = None):
    """True E, lambda and noise-free displacement at the final time of the case study."""
    cfg = cfg or CaseStudyConfig()
    pt = cfg.make_physical_twin()
    pt.measure_displacement(pt.timeframe[-1])   # appends the noise-free value
    return pt.true_E, pt.true_lambda, pt.true_displacement[-1]

def parse_results(file_path, deterioration_times):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract method name from path
    parts = file_path.split(os.sep)
    method = parts[-2]
    if method == "Sequential Updating with Physics Based Model" or method == "CD_Sequential Updating with Physics Based Model":
        method = "Seq PM"
    elif method == "Sequential Updating with SRG Prediction and Active Learning" or method == "CD_Sequential Updating with SRG Prediction and Active Learning":
        method = "Seq SRG"
        # return None
    elif method == "Sequential Updating with SRG Prediction and without Active Learning" or method == "CD_Sequential Updating with SRG Prediction and without Active Learning":
        method = "Seq SRG"
        # return None
    elif method == "Full Updating with Physics Based Model" or method == "CD_Full Updating with Physics Based Model":
        method = "Batch PM"
        # return None
    elif method == "Full Updating with SRG Prediction and Active Learning" or method == "CD_Full Updating with SRG Prediction and Active Learning":
        method = "Batch SRG"
        # return None
    elif method == "Full Updating with SRG Prediction and without Active Learning" or method == "CD_Full Updating with SRG Prediction and without Active Learning":
        method = "Batch SRG"
        # return None
    else:
        raise ValueError(f"Invalid method: {method}")

    # Find the final timestep
    timesteps = sorted(int(k) for k in data.keys() if k.isdigit())
    final_timestep = max(timesteps)
    final_sublevel = max([int(k) for k in data[str(final_timestep)].keys() if k.isdigit()])
    
    prior_state = data[str(timesteps[0])][str(0)]
    final_state = data[str(final_timestep)][str(final_sublevel)]



    for timestep in timesteps:
        for sublevel in data[str(timestep)].keys():
            displacement_0 = data[str(timestep)][sublevel]['displacement']['samples']
            displacement_0 = np.array(displacement_0).squeeze()


    # # if prior_state["displacement"]["samples"] is shape 1,1000,1 transform to 1000,



    results = {
        "Method": method,
        # "E_mu_posterior": final_state["E"]["mu"],
        # "E_sigma_posterior": final_state["E"]["sigma"],
        "E_samples_posterior": final_state["E"]["samples"],
        # "lambda_mu_posterior": final_state["lambda"]["mu"],
        # "lambda_sigma_posterior": final_state["lambda"]["sigma"],
        "lambda_samples_posterior": final_state["lambda"]["samples"],
        "I_samples_posterior": final_state["I"]["samples"],

        # Surrogate AL training set (E, I) + seed size, when present (SRG methods only;
        # None for physics-based methods, and absent in runs predating this field).
        "surrogate_train_X": final_state.get("surrogate_train_X"),
        "surrogate_n_init_train": int(final_state.get("surrogate_n_init_train") or 0),

        # "Displacement_mu_posterior": final_state["displacement"]["mu"],
        # "Displacement_sigma_posterior": final_state["displacement"]["sigma"],
        "Displacement_samples_posterior": final_state["displacement"]["samples"] if isinstance(final_state["displacement"]["samples"], list) else [final_state["displacement"]["samples"]],

        "E_mu_prior": prior_state["E"]["mu"],
        "E_sigma_prior": prior_state["E"]["sigma"],
        "E_samples_prior": prior_state["E"]["samples"],

        "lambda_mu_prior": prior_state["lambda"]["mu"],
        "lambda_sigma_prior": prior_state["lambda"]["sigma"],
        "lambda_samples_prior": prior_state["lambda"]["samples"],

        "Displacement_mu_prior": prior_state["displacement"]["mu"],
        "Displacement_sigma_prior": prior_state["displacement"]["sigma"],
        # "Displacement_samples_prior": prior_state["displacement"]["samples"] if isinstance(prior_state["displacement"]["samples"], list) else [prior_state["displacement"]["samples"]],
        "Displacement_samples_prior": np.array(prior_state["displacement"]["samples"],).squeeze().tolist(),

        # cumulative over the whole time frame, as in the app and the paper
        "Likelihood Calls": float(cumulative_calls(
            method, timesteps,
            [data[str(t)][str(max(int(k) for k in data[str(t)] if k.isdigit()))]["nr_likelihood_calls"]
             for t in timesteps],
            deterioration_times)[-1]),

        # Data this run saw (priors + full measurement record), for the reference posterior.
        "measurements": extract_measurements(data),
        "file": file_path,
    }
    results["measurement_key"] = measurement_key(results["measurements"])
    return results


def _style_violin_parts(violin_parts, facecolor, alpha=0.2):
    for body in violin_parts.get('bodies', []):
        body.set_facecolor(facecolor)
        body.set_alpha(alpha)

    for part_name in ("cbars", "cmins", "cmaxes","cmeans"):
        part = violin_parts.get(part_name)
        if part is None:
            continue
        if isinstance(part, list):
            for artist in part:
                if hasattr(artist, "set_color"):
                    artist.set_color(facecolor)
                if hasattr(artist, "set_edgecolor"):
                    artist.set_edgecolor("none")
                if hasattr(artist, "set_linewidth"):
                    artist.set_linewidth(2)
        else:
            if hasattr(part, "set_color"):
                part.set_color(facecolor)
            if hasattr(part, "set_edgecolor"):
                part.set_edgecolor("none")
            if hasattr(part, "set_linewidth"):
                part.set_linewidth(2)

REFERENCE_METHODS = ("Reference",)
# Column order of the final comparison plot: exact reference first, then the methods.
METHOD_ORDER = ["Reference", "Seq PM", "Seq SRG", "Batch PM", "Batch SRG"]
METRIC_COLUMNS = ["E_mean_err", "E_std_ratio", "E_w1",
                  "lambda_mean_err", "lambda_std_ratio", "lambda_w1",
                  "displacement_mean_err", "displacement_std_ratio", "displacement_w1",
                  "joint_energy"]


def compute_references(df, beam_type="cantilever", mcs_n=2_000_000, grid_n=400, cfg: CaseStudyConfig = None):
    """One MCS + one grid reference per distinct data set (runs of the same
    run share their measurements, so they share a reference)."""
    cfg = cfg or CaseStudyConfig(beam_type=beam_type)
    ref_kw = dict(beam_type=beam_type, beam_kwargs=cfg.beam_kwargs, t_pred=cfg.t_end)
    refs = {}
    for key, meas in df.groupby("measurement_key", sort=False)["measurements"].first().items():
        grid = reference_grid(meas, n=grid_n, **ref_kw)
        mcs = reference_mcs(meas, n=mcs_n, **ref_kw)
        refs[key] = {"grid": grid, "mcs": mcs}
        print(f"[reference {len(refs)}] E={grid['E_mean']:.4g}±{grid['E_std']:.3g}  "
              f"lambda={grid['lambda_mean']:.4g}±{grid['lambda_std']:.3g}  "
              f"MCS ESS={mcs['ess']:.0f}  "
              f"logZ grid/MCS={grid['log_evidence']:.3f}/{mcs['log_evidence']:.3f}")
        if mcs["ess"] < 1e4:
            print(f"  WARNING: low MCS ESS ({mcs['ess']:.0f}); increase --ref-mcs-n")
    return refs


def reference_metrics_table(df, refs):
    """Per-run accuracy against the grid reference (the noise-free baseline), plus
    one 'Ref MCS' row per data set so the MCS reference is scored the same way."""
    rep_id = {key: i for i, key in enumerate(refs)}
    rows = []
    for _, run in df.iterrows():
        grid = refs[run["measurement_key"]]["grid"]
        m = compare_to_reference(run["E_samples_posterior"], run["lambda_samples_posterior"],
                                 run["Displacement_samples_posterior"], grid, grid_ref=grid)
        rows.append({"Method": run["Method"], "repetition": rep_id[run["measurement_key"]],
                     "Likelihood Calls": run["Likelihood Calls"],
                     "n_samples": len(np.ravel(run["E_samples_posterior"])), **m})
    # MCS reference draws, subsampled to each sample size the methods keep: the
    # distance a perfect sampler would still show with that many samples.
    sizes = sorted({r["n_samples"] for r in rows})
    for key, ref in refs.items():
        mcs, grid = ref["mcs"], ref["grid"]
        for n in sizes:
            k = min(n, len(mcs["E_samples"]))
            m = compare_to_reference(mcs["E_samples"][:k], mcs["lambda_samples"][:k],
                                     mcs["displacement_samples"][:k], grid, grid_ref=grid)
            rows.append({"Method": f"Ref MCS n={k}", "repetition": rep_id[key],
                         "Likelihood Calls": float(mcs["n"]), "n_samples": k,
                         "mcs_ess": mcs["ess"],
                         "log_evidence_err": mcs["log_evidence"] - grid["log_evidence"], **m})
    return pd.DataFrame(rows)


def reference_summary_table(metrics_df):
    """Mean/std of each metric per method, with one 'Reference' row first.

    The Reference row is the exact grid posterior scored against itself — zero
    error, sd ratio 1 — i.e. the target every method is measured against. The
    'Ref MCS n=k' sample-size floors stay in the per-run CSV and metrics plot.
    """
    methods = metrics_df[~metrics_df["Method"].str.startswith("Ref")]
    summary = methods.groupby("Method")[METRIC_COLUMNS].agg(["mean", "std"])
    exact = {c: (1.0 if c.endswith("_std_ratio") else 0.0) for c in METRIC_COLUMNS}
    summary.loc["Reference"] = [v for c in METRIC_COLUMNS for v in (exact[c], 0.0)]
    return summary.reindex([m for m in METHOD_ORDER if m in summary.index])


def reference_sample_rows(df, refs):
    """Pooled grid-reference posterior draws (one set per data set), shaped like a
    `samples_df` row so the reference sits next to the methods in the violin plots."""
    prior_src = df.iloc[0]
    pool = lambda q: np.concatenate([r["grid"][q] for r in refs.values()]).tolist()
    row = {
        "E_samples_posterior": pool("E_samples"),
        "lambda_samples_posterior": pool("lambda_samples"),
        "Displacement_samples_posterior": pool("displacement_samples"),
        "Likelihood Calls": [],
        "E_samples_prior": prior_src["E_samples_prior"],
        "lambda_samples_prior": prior_src["lambda_samples_prior"],
        "Displacement_samples_prior": prior_src["Displacement_samples_prior"],
    }
    return pd.DataFrame.from_dict({"Reference": row}, orient="index")


def plot_reference_metrics(metrics_df, results_dir: str):
    """Distance of each method's posterior to the grid reference, one point per run.

    Panels: 1-Wasserstein / sd_ref for E, lambda, displacement, and the joint (E,
    lambda) energy distance. 'Ref MCS n=k' is the MCS reference subsampled to k
    draws and scored the same way — the floor finite sampling alone sets for a
    method keeping k samples. Lower is better; 0 is exact.
    """
    set_academic_style()
    present = list(dict.fromkeys(metrics_df["Method"]))
    methods = ([m for m in ["Seq PM", "Seq SRG", "Batch PM", "Batch SRG"] if m in present]
               + [m for m in present if m.startswith("Ref MCS")])
    panels = [("E_w1", r"$W_1(E)\,/\,\sigma_{ref}$"),
              ("lambda_w1", r"$W_1(\lambda)\,/\,\sigma_{ref}$"),
              ("displacement_w1", r"$W_1(\delta)\,/\,\sigma_{ref}$"),
              ("joint_energy", r"Energy distance $(E,\lambda)$")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8), constrained_layout=True)
    for ax, (col, label) in zip(axes, panels):
        for i, m in enumerate(methods, start=1):
            v = metrics_df.loc[metrics_df["Method"] == m, col].to_numpy(dtype=float)
            color = PRIOR_COLOR if m.startswith("Ref") else PRED_COLOR
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=22, color=color,
                       alpha=0.6, edgecolors="none")
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=TRUE_COLOR, linewidth=2)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distance to reference posterior (one point per run, bar = median)")
    plot_dir = f"{results_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, "reference_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    # Accuracy vs cost: one point per run, model calls on a log axis.
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    markers = dict(zip(["Seq PM", "Seq SRG", "Batch PM", "Batch SRG"], ["o", "s", "^", "D"]))
    for m, mk in markers.items():
        sub = metrics_df[metrics_df["Method"] == m]
        if len(sub):
            ax.scatter(sub["Likelihood Calls"], sub["joint_energy"], marker=mk, s=40,
                       color=PRED_COLOR if "SRG" in m else TRUE_COLOR, alpha=0.7, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Model calls")
    ax.set_ylabel(r"Energy distance to reference $(E,\lambda)$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(plot_dir, "reference_accuracy_vs_cost.png"), bbox_inches="tight")
    plt.close(fig)


def plot_results(df, true_E, true_lambda, true_displacement, results_dir: str = "results",
                 reference_density: dict = None):
    """Violin comparison of all methods. ``reference_density`` maps 'E', 'lambda',
    'displacement' to ``(y, density, mean)`` of the exact reference posterior; the
    'Reference' column is then drawn from that density instead of a sample KDE."""
    sns.set_theme(style="whitegrid")
    set_academic_style()

    methods = df.index

    other_fontsize = 30
    legend_fontsize = 24

    def _scalarize(value, name):
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"Expected {name} to be a scalar, but got shape {arr.shape}")
        return float(arr.reshape(-1)[0])

    true_E_value = _scalarize(true_E, "true_E")
    true_lambda_value = _scalarize(true_lambda, "true_lambda")
    true_displacement_value = _scalarize(true_displacement, "true_displacement")

    prior_color = PRIOR_COLOR
    prior_alpha = 0.2
    posterior_color = PRED_COLOR
    colors = posterior_color
    posterior_alpha = 0.5

    def _posterior_violins(ax, column, quantity):
        """Sample-KDE violins for the methods; exact-density violin for the reference."""
        positions = [i for i, m in enumerate(methods, start=1)
                     if not (m in REFERENCE_METHODS and reference_density)]
        violins = ax.violinplot([df[column][methods[i - 1]] for i in positions],
                                positions=positions, showmeans=True)
        _style_violin_parts(violins, colors, alpha=posterior_alpha)
        if not reference_density:
            return
        y, dens, mean = reference_density[quantity]
        half = 0.25 * dens / dens.max()           # same max half-width as violinplot's default
        support = y[dens > 1e-3 * dens.max()]
        y, half = y[(y >= support[0]) & (y <= support[-1])], half[(y >= support[0]) & (y <= support[-1])]
        for i, m in enumerate(methods, start=1):
            if m in REFERENCE_METHODS:
                ax.fill_betweenx(y, i - half, i + half, color=colors, alpha=posterior_alpha,
                                 linewidth=0)
                ax.hlines(mean, i - 0.12, i + 0.12, color=colors, linewidth=2)

    # --- Plotting Functions ---

    def plot_e_modulus(ax):
        e_samples_prior = df['E_samples_prior']
        ax.axhline(y=true_E_value, color=TRUE_COLOR, linestyle='--', label=f'True E = {true_E_value:.2e}', linewidth=2)
        prior_violin = ax.violinplot(e_samples_prior, showmeans=True)
        _style_violin_parts(prior_violin, prior_color, alpha=prior_alpha)
        _posterior_violins(ax, 'E_samples_posterior', 'E')
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.set_ylabel(r'E [$kN/m^2$]', fontsize=other_fontsize)
        ax.tick_params(axis='x', labelsize=other_fontsize)
        ax.bar(0, 0, color=prior_color, alpha=prior_alpha, label='Prior')
        ax.bar(0, 0, color=posterior_color, alpha=posterior_alpha, label='Posterior')
        ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)

    def plot_lambda(ax):
        lambda_samples_prior = df['lambda_samples_prior']
        lambda_prior_violin = ax.violinplot(lambda_samples_prior, showmeans=True)
        _style_violin_parts(lambda_prior_violin, prior_color, alpha=prior_alpha)
        ax.axhline(y=true_lambda_value, color=TRUE_COLOR, linestyle='--', label=f'True lambda = {true_lambda_value:.2f}', linewidth=2)
        _posterior_violins(ax, 'lambda_samples_posterior', 'lambda')
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.set_ylim(0, 2.5)
        ax.set_ylabel(r'$\lambda$ [\%]', fontsize=other_fontsize)
        ax.tick_params(axis='x', labelsize=other_fontsize)
        ax.bar(0, 0, color=prior_color, alpha=prior_alpha, label='Prior')
        ax.bar(0, 0, color=posterior_color, alpha=posterior_alpha, label='Posterior')
        ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)

    def plot_displacement(ax):
        # get samples for Seq PM
        seq_pm_prior_displacement_samples = df['Displacement_samples_prior']["Seq PM"]
        # Ensure it is a flat list of floats
        seq_pm_prior_displacement_samples = np.array(seq_pm_prior_displacement_samples).flatten().tolist()
        # create a list with the length of the methods, repeating the samples for Seq PM
        seq_pm_prior_displacement_samples = [seq_pm_prior_displacement_samples for _ in methods]



        prior_displacement_violin = ax.violinplot(seq_pm_prior_displacement_samples, showmeans=True)
        _style_violin_parts(prior_displacement_violin, prior_color, alpha=prior_alpha)
        _posterior_violins(ax, 'Displacement_samples_posterior', 'displacement')
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.axhline(y=true_displacement_value, color=TRUE_COLOR, linestyle='--', label=f'True displacement = {true_displacement_value:.2f}', linewidth=2)
        ax.set_ylabel('Displacement [m]', fontsize=other_fontsize)
        ax.tick_params(axis='x', labelsize=other_fontsize)
        # set y-axis length
        ax.set_ylim(0, 0.5)
        ax.bar(0, 0, color=posterior_color, alpha=posterior_alpha, label='Posterior')
        ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)

    def plot_likelihood(ax):
        # Keep columns aligned with the other panels; the reference has no
        # comparable call count, so its column is marked n/a.
        positions = [i for i, m in enumerate(methods, start=1) if m not in REFERENCE_METHODS]
        likelihood_calls = [np.asarray(df['Likelihood Calls'][methods[i - 1]]) for i in positions]
        violins = ax.violinplot(likelihood_calls, positions=positions, showmeans=True, widths=0.8)
        # style the violin parts
        _style_violin_parts(violins, colors , alpha=posterior_alpha)

        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels(methods)
        ax.set_xlim(0.5, len(methods) + 0.5)
        ax.set_yscale('log')
        ax.set_ylabel('Model Calls', fontsize=other_fontsize)
        ax.tick_params(axis='x', labelsize=other_fontsize)

        for i, m in enumerate(methods, start=1):
            if m in REFERENCE_METHODS:
                ax.text(i, 0.5, 'n/a', transform=ax.get_xaxis_transform(), ha='center',
                        va='center', fontsize=other_fontsize, color='0.5')

        likelihood_means = [np.mean(calls) for calls in likelihood_calls]
        for idx, mean in zip(positions, likelihood_means):
            ax.text(
                idx,
                mean,
                f'{mean:.0f}',
                ha='center',
                va='bottom',
                fontsize=other_fontsize,
                rotation=0
            )

        # ax.legend(handles=[
        # ], fontsize=legend_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=legend_fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=legend_fontsize)

    # Create plot directory if it doesn't exist
    plot_dir = f'{results_dir}/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- Combined Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(25, 12))
    
    plot_e_modulus(axes[0,0])
    plot_lambda(axes[0,1])
    plot_displacement(axes[1,0])
    plot_likelihood(axes[1,1])

    # Add borders around each subplot for clear differentiation
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color('0.3')

    legend_below(fig, ncol=5, fontsize=legend_fontsize)
    fig.savefig(os.path.join(plot_dir, 'comparison_plot.png'), bbox_inches='tight')
    plt.close(fig)

    # --- Individual Plots ---
    fig_width = 9
    fig_height = 12

    def _add_borders(fig):
        for ax in fig.get_axes():
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color('0.3')

    # E Modulus
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_e_modulus(ax)
    _add_borders(fig)
    legend_below(fig, fontsize=legend_fontsize)
    fig.savefig(os.path.join(plot_dir, 'e_modulus_plot.png'), bbox_inches='tight')
    plt.close(fig)

    # Lambda
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_lambda(ax)
    _add_borders(fig)
    legend_below(fig, fontsize=legend_fontsize)
    fig.savefig(os.path.join(plot_dir, 'lambda_plot.png'), bbox_inches='tight')
    plt.close(fig)

    # Displacement
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_displacement(ax)
    _add_borders(fig)
    legend_below(fig, fontsize=legend_fontsize)
    fig.savefig(os.path.join(plot_dir, 'displacement_plot.png'), bbox_inches='tight')
    plt.close(fig)

    # Likelihood
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_likelihood(ax)
    _add_borders(fig)
    legend_below(fig, fontsize=legend_fontsize)
    fig.savefig(os.path.join(plot_dir, 'likelihood_plot.png'), bbox_inches='tight')
    plt.close(fig)

def plot_acquisition_storyboard(df, true_displacement, results_dir: str,
                                true_E=None, l: float = 10.0, V: float = 4.0):
    """Problem-storyboard-style panels of WHICH (E, I) points each method selected.

    Every panel shares the same backdrop — the pooled posterior (E, I) cloud across
    all methods plus the measurement limit-state contour E·I = V·l⁴/(8·y) — so the
    only thing that changes panel-to-panel is the method's own surrogate AL design.
    The initial seed is drawn faint; AL-added points are highlighted and counted.

    Reads the per-method ``surrogate_train_X`` carried in the state-dict JSONs.
    Methods without it (physics-based, or runs predating the field) render as 'n/a'.
    """
    set_academic_style()

    true_disp = float(np.asarray(true_displacement).reshape(-1)[0])
    # True (E, I): ground-truth E with I on the (final) measurement LSF at that E.
    E_true = float(np.asarray(true_E).reshape(-1)[0]) if true_E is not None else None
    I_true = ((V * l ** 4) / (8.0 * true_disp * E_true)
              if (E_true and true_disp > 0) else None)

    # One representative (first non-null) training set per method.
    train_by_method, ninit_by_method = {}, {}
    for method, row in df.iterrows():
        tX = row.get('surrogate_train_X')
        if tX is not None and len(tX) and method not in train_by_method:
            train_by_method[method] = np.asarray(tX, dtype=float)
            ninit_by_method[method] = int(row.get('surrogate_n_init_train') or 0)

    # Shared backdrop cloud: pooled posterior (E, I) across every method.
    E_pool, I_pool = [], []
    for _, row in df.iterrows():
        E_pool.append(np.asarray(row['E_samples_posterior'], dtype=float).ravel())
        I_pool.append(np.asarray(row['I_samples_posterior'], dtype=float).ravel())
    E_pool = np.concatenate(E_pool) if E_pool else np.array([])
    I_pool = np.concatenate(I_pool) if I_pool else np.array([])

    methods = list(df.index)
    if not train_by_method:
        print("[storyboard] no surrogate_train_X found in any run — "
              "re-run to populate this field. Skipping storyboard.")
        return

    # Axis limits from the cloud, expanded to contain every method's points.
    E_lo, E_hi = float(E_pool.min()), float(E_pool.max())
    I_lo, I_hi = float(I_pool.min()), float(I_pool.max())
    for tX in train_by_method.values():
        E_lo, E_hi = min(E_lo, tX[:, 0].min()), max(E_hi, tX[:, 0].max())
        I_lo, I_hi = min(I_lo, tX[:, 1].min()), max(I_hi, tX[:, 1].max())
    if I_true is not None:
        E_lo, E_hi = min(E_lo, E_true), max(E_hi, E_true)
        I_lo, I_hi = min(I_lo, I_true), max(I_hi, I_true)
    padE, padI = 0.05 * (E_hi - E_lo), 0.05 * (I_hi - I_lo)
    xlim = (E_lo - padE, E_hi + padE)
    ylim = (max(I_lo - padI, 1e-9 * max(I_hi, 1.0)), I_hi + padI)
    Egrid = np.linspace(xlim[0], xlim[1], 200)

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    fig.suptitle('Active-learning comparison — point selection per method')
    for k, method in enumerate(methods):
        ax = axes[0][k]
        ax.scatter(E_pool, I_pool, s=4, color=PRIOR_COLOR, alpha=0.35,
                   edgecolors='none', zorder=0)
        if true_disp > 0:
            ax.plot(Egrid, (V * l ** 4) / (8.0 * true_disp * Egrid),
                    color=MEAS_COLOR, lw=1.4, alpha=0.7, zorder=1,
                    label='measurement LSF')
        if I_true is not None:
            ax.scatter([E_true], [I_true], marker='*', s=240, c=TRUTH_COLOR,
                       edgecolors='black', linewidths=0.8, zorder=6)
        title = str(method)
        if method in train_by_method:
            tX = train_by_method[method]
            n0 = ninit_by_method[method]
            seed, added = tX[:n0], tX[n0:]
            ax.scatter(seed[:, 0], seed[:, 1], marker='^', s=40, c=TRUE_COLOR,
                       edgecolors='white', linewidths=0.4, alpha=0.6, zorder=4)
            ax.scatter(added[:, 0], added[:, 1], marker='o', s=48, c=PRED_COLOR,
                       edgecolors='black', linewidths=0.5, zorder=5)
            title += f"  (+{len(added)} pts)"
        else:
            ax.text(0.5, 0.5, 'no surrogate / n/a', transform=ax.transAxes,
                    ha='center', va='center', color='0.55', fontsize='small')
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax.set_xlabel(r'$E$ [kN/m$^2$]')
        if k == 0:
            ax.set_ylabel(r'$I$ [m$^4$]')

    handles = [
        mpatches.Patch(color=PRIOR_COLOR, alpha=0.5, label='posterior cloud (shared)'),
        plt.Line2D([0], [0], color=MEAS_COLOR, lw=1.6,
                   label=r'measurement LSF  $E\,I = V l^4/(8\,y)$'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=TRUE_COLOR,
                   markersize=10, label='initial seed design'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PRED_COLOR,
                   markeredgecolor='black', markersize=10,
                   label='AL-selected points (+N)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=TRUTH_COLOR,
                   markeredgecolor='black', markersize=14, label='true (E, I)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.13))

    plot_dir = f'{results_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, 'acquisition_storyboard.png'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved acquisition storyboard to {plot_dir}/acquisition_storyboard.png")


def main(results_folder: str = "results/case_study", beam_type: str = "cantilever",
         storyboard: bool = False, reference: bool = True,
         ref_mcs_n: int = 2_000_000, ref_grid_n: int = 400):
    cfg = CaseStudyConfig.from_results_folder(results_folder, beam_type=beam_type)
    results_dir = f'{results_folder}/{beam_type}'
    json_files = find_json_files(results_dir)
    
    if not json_files:
        print("No result files found.")
        return

    all_results = [r for r in (parse_results(file, cfg.deterioration_times) for file in json_files) if r is not None]

    df = pd.DataFrame(all_results)
    # print all columns 
    pd.set_option("display.max_columns", None)
    print(df.head())

    
    def list_mean(series):
        """Flatten all lists in series and compute mean"""
        all_values = np.concatenate([np.asarray(v, dtype=float).ravel() for v in series.values])
        return np.mean(all_values)

    def list_std(series):
        """Flatten all lists in series and compute std"""
        all_values = np.concatenate([np.asarray(v, dtype=float).ravel() for v in series.values])
        return np.std(all_values)

    # Use custom functions for list columns, regular mean/std for scalar columns
    summary_df = df.groupby('Method').agg({
        'E_samples_posterior': [list_mean, list_std],
        'lambda_samples_posterior': [list_mean, list_std],
        'Displacement_samples_posterior': [list_mean, list_std],
        'Likelihood Calls': ['mean', 'std'],  # if this is scalar
        'E_samples_prior': [list_mean, list_std],
        'lambda_samples_prior': [list_mean, list_std],
        'Displacement_samples_prior': [list_mean, list_std],
                            })
    # Group by method and calculate mean and std for metrics
    # summary_df = df.groupby('Method').agg({
    #     # 'E_mu_posterior': ['mean', 'std'],
    #     # 'E_sigma_posterior': ['mean', 'std'],
    #     'E_samples_posterior': ['mean', 'std'],
    #     'lambda_samples_posterior': ['mean', 'std'],
    #     'Displacement_samples_posterior': ['mean', 'std'],
    #     'Likelihood Calls': ['mean', 'std'],
    #     'E_samples_prior': ['mean', 'std'],
    #     'lambda_samples_prior': ['mean', 'std'],
    #     'Displacement_samples_prior': ['mean', 'std'],
    # })
    print("--- Results Summary ---")
    pd.set_option("display.max_columns", None)
    print(summary_df)

    
    true_E, true_lambda, true_displacement = get_true_values(cfg)
    print(f"true_E: {true_E}, true_lambda: {true_lambda}, true_displacement: {true_displacement}")
    
    # Group by method and aggregate samples
    samples_df = df.groupby('Method').agg({
        'E_samples_posterior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
        'lambda_samples_posterior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
        'Displacement_samples_posterior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
        'Likelihood Calls': lambda x: x.tolist(),
        'E_samples_prior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
        'lambda_samples_prior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
        'Displacement_samples_prior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
    })




    # Prepare DataFrame for plotting
    # plot_df = df.groupby('Method').agg({
    #     'E_mu': 'mean',
    #     'E_sigma': 'mean',
    #     'lambda_mu': 'mean',
    #     'lambda_sigma': 'mean',
    #     'Likelihood Calls': 'mean',
    # })
    
    if reference:
        # Exact-posterior baseline per data set: grid quadrature (scoring baseline)
        # cross-checked by prior Monte Carlo; both are also drawn in the violins.
        refs = compute_references(df, beam_type=beam_type, mcs_n=ref_mcs_n, grid_n=ref_grid_n, cfg=cfg)
        metrics_df = reference_metrics_table(df, refs)
        os.makedirs(f"{results_folder}/plots", exist_ok=True)
        metrics_df.to_csv(f"{results_folder}/plots/reference_metrics.csv", index=False)
        metric_summary = reference_summary_table(metrics_df)
        print("--- Accuracy vs reference posterior (grid) ---")
        print(metric_summary.T.to_string(float_format=lambda v: f"{v:.3f}"))
        metric_summary.to_csv(f"{results_folder}/plots/reference_metrics_summary.csv")
        plot_reference_metrics(metrics_df, results_dir=results_folder)
        samples_df = pd.concat([reference_sample_rows(samples_df, refs), samples_df])

    samples_df = samples_df.reindex([m for m in METHOD_ORDER if m in samples_df.index]
                                    + [m for m in samples_df.index if m not in METHOD_ORDER])

    reference_density = None
    if reference:
        grids = [r["grid"] for r in refs.values()]
        reference_density = {q: pooled_marginal_density(grids, q) for q in ("E", "lambda", "displacement")}

    plot_results(samples_df, true_E, true_lambda, true_displacement,
                 results_dir=results_folder, reference_density=reference_density)

    if storyboard:
        # Method-indexed frame retaining the (E, I) cloud + surrogate training set.
        story_df = df.groupby('Method').agg({
            'E_samples_posterior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
            'I_samples_posterior': lambda x: np.concatenate([np.asarray(v, dtype=float).ravel() for v in x]).tolist(),
            'surrogate_train_X': lambda x: next((v for v in x if v is not None and len(v)), None),
            'surrogate_n_init_train': 'max',
        })
        plot_acquisition_storyboard(story_df, true_displacement, results_dir=results_folder,
                                    true_E=true_E)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Paper, Section 5.2.3: compare the updating methods against the exact posterior")
    p.add_argument("--results-folder", default="results/case_study")
    p.add_argument("--beam-type", default="cantilever")
    p.add_argument("--storyboard", action="store_true",
                   help="also render the AL point-selection storyboard "
                        "(shared (E, I) cloud + measurement LSF, per-method selections)")
    p.add_argument("--presentation", action="store_true",
                   help="Arial / larger-font presentation styling for all plots")
    p.add_argument("--no-reference", action="store_true",
                   help="skip the exact reference posterior (grid quadrature + MCS)")
    p.add_argument("--ref-mcs-n", type=int, default=2_000_000,
                   help="prior samples for the MCS reference (per data set)")
    p.add_argument("--ref-grid-n", type=int, default=400,
                   help="grid nodes per axis for the quadrature reference")
    a = p.parse_args()
    if a.presentation:
        os.environ["PLOT_PRESENTATION"] = "1"   # read by set_academic_style() in every plot
    main(results_folder=a.results_folder, beam_type=a.beam_type, storyboard=a.storyboard,
         reference=not a.no_reference, ref_mcs_n=a.ref_mcs_n, ref_grid_n=a.ref_grid_n)