import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Union

from utils.plot_style import set_academic_style, legend_below, MEAS_COLOR


def plot_scenario_comparison(
    pdts,
    methods_names,
    physical_twin,
    timeframe: np.ndarray,
    plot_dir: Union[str, Path, None] = None,
    violin_plot: bool = False,
    hide_t0_displacement: Tuple[str, ...] = (),
) -> None:
    """
    Compare the evolution of four digital twin approaches against the physical twin.

    ``hide_t0_displacement``: method names whose t=0 displacement is not drawn, e.g. a
    surrogate's prediction before any active-learning training, which is not meaningful.
    """

    sns.set_theme(style="whitegrid")
    set_academic_style()

    # Presentation-friendly typography (Arial, larger fonts for PowerPoint).
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 15,
        "figure.titlesize": 22,
    })

    methods = [(methods_names[i], pdts[i]) for i in range(len(pdts))]
    colors = sns.color_palette("husl", len(methods)+1)
    colors = colors[1:]

    # Collect all timesteps present in the timeframe or any state_dict.
    timestep_set = set(float(t) for t in np.atleast_1d(timeframe))
    for _, pdt in methods:
        timestep_set.update(float(k) for k in pdt.state_dict.keys())
    timesteps = np.array(sorted(timestep_set), dtype=float)

    if len(timesteps) > 1:
        base_diffs = np.diff(timesteps)
        base_step = float(np.min(base_diffs[base_diffs > 0])) if np.any(base_diffs > 0) else 1.0
    else:
        base_step = 1.0

    max_sublevels = 1
    for _, pdt in methods:
        for sublevels in pdt.state_dict.values():
            max_sublevels = max(max_sublevels, len(sublevels))
    sublevel_spacing = base_step / (max_sublevels + 1)
    if sublevel_spacing <= 0:
        sublevel_spacing = 0.25

    def _collect_series(pdt) -> Dict[str, Dict[str, Union[np.ndarray, list]]]:
        series = {metric: {"times": [], "mu": [], "sigma": [], "samples": []} for metric in ("lambda", "displacement", "I", "E")}
        sorted_times = sorted(pdt.state_dict.keys(), key=lambda x: float(x))
        times_to_extract = sorted(list(set(list(sorted_times) + [0])))
        for t in times_to_extract:
            sublevels = pdt.state_dict[t]
            sorted_keys = sorted(sublevels.keys())
            if not sorted_keys:
                continue

            sub_key = sorted_keys[-1]
            state = sublevels[sub_key]
            time_point = float(t)

            for metric in ("lambda", "displacement", "I"):
                series[metric]["times"].append(time_point)
                series[metric]["mu"].append(float(state[metric]["mu"]))
                series[metric]["sigma"].append(float(state[metric]["sigma"]))
                series[metric]["samples"].append(state[metric]["samples"])
            series["E"]["times"].append(time_point)
            series["E"]["mu"].append(float(state["E"]["mu"]))
            series["E"]["sigma"].append(float(state["E"]["sigma"]))
            series["E"]["samples"].append(state["E"]["samples"])
        for metric in series:
            series[metric]["times"] = np.array(series[metric]["times"], dtype=int)
            series[metric]["mu"] = np.array(series[metric]["mu"], dtype=float)
            series[metric]["sigma"] = np.array(series[metric]["sigma"], dtype=float)
        return series

    def _plot_series_boxplot(ax, times: np.ndarray, samples: list, color: Tuple[float, float, float], label: str) -> None:
        if not samples:
            return

        valid_indices = [i for i, s in enumerate(samples) if s is not None and len(s) > 0]
        if not valid_indices:
            return

        valid_times = times[valid_indices]
        valid_samples = [np.array(samples[i]).flatten() for i in valid_indices]

        if valid_times.size > 1:
            sorted_unique = np.unique(np.sort(valid_times))
            diffs = np.diff(sorted_unique)
            min_spacing = diffs[diffs > 0].min() if np.any(diffs > 0) else 1.0
        else:
            min_spacing = 1.0
        box_width = 0.45 * min_spacing

        boxprops = dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5)
        medianprops = dict(color=color, linewidth=2)
        whiskerprops = dict(color=color, linewidth=1.5)
        capprops = dict(color=color, linewidth=1.5)
        meanprops = dict(marker="o", markerfacecolor=color, markeredgecolor="white", markersize=5)

        means = [np.mean(s) for s in valid_samples]
        ax.plot(valid_times, means, "-o", color=color, label=label)

        ax.boxplot(
            valid_samples,
            positions=valid_times,
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            meanprops=meanprops,
            showfliers=False,
        )


    def _plot_series_violin(ax, times: np.ndarray, samples: list, color: Tuple[float, float, float], label: str) -> None:
        if not samples:
            return

        valid_indices = [i for i, s in enumerate(samples) if s is not None and len(s) > 0]
        if not valid_indices:
            return

        valid_times = times[valid_indices]
        valid_samples = [np.array(samples[i]).flatten() for i in valid_indices]

        if valid_times.size > 1:
            sorted_unique = np.unique(np.sort(valid_times))
            diffs = np.diff(sorted_unique)
            min_spacing = diffs[diffs > 0].min() if np.any(diffs > 0) else 1.0
        else:
            min_spacing = 1.0
        violin_width = 0.6 * min_spacing

        parts = ax.violinplot(
            valid_samples,
            positions=valid_times,
            widths=violin_width,
            showmeans=True,
            showextrema=True,
            showmedians=True,
        )

        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)

            path = body.get_paths()[0]
            verts = path.vertices
            center = valid_times[i]
            verts[:, 0] = np.clip(verts[:, 0], center, np.inf)

        for key in ("cmins", "cmaxes", "cmeans", "cmedians", "cbars"):
            if key in parts:
                lc = parts[key]
                lc.set_color(color)
                if key != "cbars":
                    segments = lc.get_segments()
                    new_segments = []
                    for i, seg in enumerate(segments):
                        center = valid_times[i]
                        seg[:, 0] = np.clip(seg[:, 0], center, np.inf)
                        new_segments.append(seg)
                    lc.set_segments(new_segments)

        ax.plot([], [], color=color, label=label, linewidth=2)


    def _extract_measurements(state_dict: Dict, measurement_key: str, scale_sigma_by_mean: bool = False):
        aggregated: Dict[float, Tuple[float, float]] = {}
        for t, sublevels in state_dict.items():
            for state in sublevels.values():
                measurement = state.get(measurement_key, {})
                mean = measurement.get("mean")
                sigma = measurement.get("sigma")
                if mean is None or sigma is None:
                    continue
                time_value = measurement.get("time", t)
                means = np.atleast_1d(mean).astype(float)
                sigmas = np.atleast_1d(sigma).astype(float)
                times = np.atleast_1d(time_value).astype(float)

                if times.size == 1 and means.size > 1:
                    times = np.full_like(means, times.item(), dtype=float)
                if sigmas.size == 1 and means.size > 1:
                    sigmas = np.full_like(means, sigmas.item(), dtype=float)

                for tm, mu, sig in zip(times, means, sigmas):
                    adjusted_sigma = sig * mu if scale_sigma_by_mean else sig
                    aggregated[float(tm)] = (float(mu), float(adjusted_sigma))

        times_sorted = np.array(sorted(aggregated.keys()), dtype=float)
        if times_sorted.size == 0:
            return times_sorted, np.array([], dtype=float), np.array([], dtype=float)
        means_sorted = np.array([aggregated[t][0] for t in times_sorted], dtype=float)
        sigmas_sorted = np.array([aggregated[t][1] for t in times_sorted], dtype=float)
        return times_sorted, means_sorted, sigmas_sorted

    series_by_method = {label: _collect_series(pdt) for label, pdt in methods}
    for label in hide_t0_displacement:
        if label in series_by_method:
            disp = series_by_method[label]["displacement"]
            keep = np.where(disp["times"] != 0)[0]
            disp["times"], disp["mu"], disp["sigma"] = disp["times"][keep], disp["mu"][keep], disp["sigma"][keep]
            disp["samples"] = [disp["samples"][i] for i in keep]

    fig, axs = plt.subplots(2, 2, figsize=(12, 5), constrained_layout=True)
    ax_displacement = axs[0, 0]
    ax_lambda = axs[0, 1]
    ax_E = axs[1, 0]
    ax_I = axs[1, 1]

    for (label, pdt), color in zip(methods, colors):
        summary = series_by_method[label]
        legend_label = f"{label} ({getattr(pdt, 'nr_likelihood_calls', 'N/A')} calls)"
        if violin_plot:
            _plot_series_violin(ax_lambda, summary["lambda"]["times"], summary["lambda"]["samples"], color, legend_label)
            _plot_series_violin(ax_displacement, summary["displacement"]["times"], summary["displacement"]["samples"], color, legend_label)
            _plot_series_violin(ax_I, summary["I"]["times"], summary["I"]["samples"], color, legend_label)
            _plot_series_violin(ax_E, summary["E"]["times"], summary["E"]["samples"], color, legend_label)
        else:
            _plot_series_boxplot(ax_lambda, summary["lambda"]["times"], summary["lambda"]["samples"], color, legend_label)
            _plot_series_boxplot(ax_displacement, summary["displacement"]["times"], summary["displacement"]["samples"], color, legend_label)
            _plot_series_boxplot(ax_I, summary["I"]["times"], summary["I"]["samples"], color, legend_label)
            _plot_series_boxplot(ax_E, summary["E"]["times"], summary["E"]["samples"], color, legend_label)

    # Observations
    prop_times, prop_means, prop_sigmas = _extract_measurements(pdts[0].state_dict, "property_measurement")
    if prop_times.size > 0:
        ax_lambda.errorbar(
            prop_times, prop_means, yerr=prop_sigmas, fmt="x", color=MEAS_COLOR, label="Observation", capsize=4
        )

    disp_times, disp_means, disp_sigmas = _extract_measurements(
        pdts[0].state_dict, "behavior_measurement", scale_sigma_by_mean=False
    )
    if disp_times.size > 0:
        ax_displacement.errorbar(
            disp_times, disp_means, yerr=disp_sigmas, fmt="x", color=MEAS_COLOR, label="Observation", capsize=4
        )

    # True values from the physical twin.
    true_lambda = np.repeat(float(physical_twin.true_lambda), len(timesteps))
    true_E = np.repeat(float(physical_twin.true_E), len(timesteps))
    true_deterioration = physical_twin.displacement_pmodel.get_loss_at_time(
        t=timesteps, lambda_dr=np.repeat(float(physical_twin.true_lambda), len(timesteps))
    )
    true_I = physical_twin.displacement_pmodel.get_I_from_loss(loss=true_deterioration)
    true_I = np.array(true_I, dtype=float).reshape(-1)
    true_displacement = physical_twin.displacement_pmodel.get_beam_max_deflection(E=true_E, I=true_I)
    true_displacement = np.array(true_displacement, dtype=float).reshape(-1)

    lsf_displacement = 0.2

    ax_lambda.plot(timesteps, true_lambda, linestyle="--", color="grey", label="True Value", zorder=10)
    ax_E.axhline(true_E[0], linestyle="--", color="grey", label="True Value", zorder=10)
    ax_I.scatter(timesteps, true_I, color="grey", label="True Value", marker="x", zorder=10)
    ax_displacement.scatter(timesteps, true_displacement, color="grey", label="True Value", marker="x", zorder=10)
    ax_displacement.axhline(
        lsf_displacement,
        linestyle="--",
        color="red",
        label="Limit State Displacement (20-Year Service Life)",
        zorder=10,
    )

    # Axis formatting
    ax_lambda.set_xlabel("Time [years]")
    ax_lambda.set_ylabel("Deterioration\nRate [%/year]")
    ax_lambda.set_xlim(left=-0.5)

    ax_displacement.set_xlabel("Time [years]")
    ax_displacement.set_ylabel("Displacement\n[m]")
    ax_displacement.set_xlim(left=-0.5)

    ax_E.set_xlabel("Time [years]")
    ax_E.set_ylabel("Young's\nModulus [$kN/m^2$]")
    ax_E.set_xlim(left=-0.5)

    ax_I.set_xlabel("Time [years]")
    ax_I.set_ylabel("Moment of\nInertia [$m^4$]")
    ax_I.set_xlim(left=-0.5)

    ax_lambda.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax_E.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax_displacement.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax_I.xaxis.set_major_locator(plt.MultipleLocator(1))

    # Single legend below figure
    legend_below(fig)

    # Deduce output directory.
    if plot_dir is None:
        base_dir = getattr(pdts[0], "plot_dir", None)
        if base_dir:
            plot_dir = Path(base_dir).parent / "comparison"
        else:
            plot_dir = Path("plots") / "comparison"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    names_str = "_".join(methods_names)
    if violin_plot:
        filename = plot_dir / f"scenario_comparison_{names_str}_{time_id}_violin.png"
    else:
        filename = plot_dir / f"scenario_comparison_{names_str}_{time_id}.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_scenario_comparison_property(
    pdts,
    methods_names,
    physical_twin,
    timeframe: np.ndarray,
    measured_corrosion_times: np.ndarray,
    plot_dir: Union[str, Path, None] = None,
    prior_variants: Tuple[bool, ...] = (False, True),
) -> None:
    """
    Compare the evolution of digital twin properties (lambda, I) at measured corrosion times.
    Plots the first sublevel at each measured time using one-sided violin plots of the samples.

    One figure is saved per entry in ``prior_variants``: ``False`` gives the plain plot,
    ``True`` additionally draws the prior I (prior lambda propagated to each measured
    corrosion time) as a grey left half-violin next to each posterior (suffix ``_with_prior``).
    """
    sns.set_theme(style="whitegrid")
    set_academic_style()

    # Presentation-friendly typography (Arial, larger fonts for PowerPoint).
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 15,
        "figure.titlesize": 22,
    })

    methods = [(methods_names[i], pdts[i]) for i in range(len(pdts))]
    colors = sns.color_palette("husl", len(methods)+1)
    colors = colors[1:]

    def _collect_series(pdt) -> Dict[str, Dict[str, Union[np.ndarray, list]]]:
        series = {metric: {"times": [], "samples": []} for metric in ("lambda", "I")}

        state_dict_float_keys = {float(k): v for k, v in pdt.state_dict.items()}

        times_to_extract = sorted(list(set(list(measured_corrosion_times) + [0])))

        for t in times_to_extract:
            t_float = float(t)
            if t_float not in state_dict_float_keys:
                continue

            sublevels = state_dict_float_keys[t_float]
            sorted_keys = sorted(sublevels.keys())
            if not sorted_keys:
                continue

            sub_key = sorted_keys[0]
            state = sublevels[sub_key]
            time_point = float(t)

            for metric in ("lambda", "I"):
                if metric in state and "samples" in state[metric]:
                    samples = state[metric]["samples"]
                    if samples is not None and len(samples) > 0:
                        series[metric]["times"].append(time_point)
                        series[metric]["samples"].append(samples)

        for metric in series:
            series[metric]["times"] = np.array(series[metric]["times"], dtype=float)
        return series

    def _plot_series_violin(ax, times: np.ndarray, samples: list, color: Tuple[float, float, float], label: str,
                            side: str = "right") -> None:
        if not samples:
            return

        valid_indices = [i for i, s in enumerate(samples) if s is not None and len(s) > 0]
        if not valid_indices:
            return

        valid_times = times[valid_indices]
        valid_samples = [np.array(samples[i]).flatten() for i in valid_indices]

        # Width from the spacing of all plotted times, so prior and posterior halves match.
        all_times = np.unique(np.array(sorted(set(list(measured_corrosion_times) + [0])), dtype=float))
        diffs = np.diff(all_times)
        min_spacing = diffs[diffs > 0].min() if np.any(diffs > 0) else 1.0
        violin_width = 0.6 * min_spacing
        lo, hi = (0.0, np.inf) if side == "right" else (-np.inf, 0.0)

        parts = ax.violinplot(
            valid_samples,
            positions=valid_times,
            widths=violin_width,
            showmeans=True,
            showextrema=True,
            showmedians=True,
        )

        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)

            path = body.get_paths()[0]
            verts = path.vertices
            center = valid_times[i]
            verts[:, 0] = np.clip(verts[:, 0], center + lo, center + hi)

        for key in ("cmins", "cmaxes", "cmeans", "cmedians", "cbars"):
            if key in parts:
                lc = parts[key]
                lc.set_color(color)
                if key != "cbars":
                    segments = lc.get_segments()
                    new_segments = []
                    for i, seg in enumerate(segments):
                        center = valid_times[i]
                        seg[:, 0] = np.clip(seg[:, 0], center + lo, center + hi)
                        new_segments.append(seg)
                    lc.set_segments(new_segments)

        ax.plot([], [], color=color, label=label, linewidth=2)

    def _prior_I_series(pdt):
        """Prior I at each measured corrosion time (>0): prior lambda samples from the
        t=0 state pushed through the deterioration model at that time."""
        sd = {float(k): v for k, v in pdt.state_dict.items()}
        if 0.0 not in sd or not sd[0.0]:
            return np.array([], dtype=float), []
        prior_state = sd[0.0][sorted(sd[0.0].keys())[0]]
        lam_prior = np.asarray(prior_state["lambda"]["samples"], dtype=float).flatten()
        times = np.array(sorted(float(t) for t in measured_corrosion_times if float(t) > 0), dtype=float)
        samples = [np.asarray(pdt.displacement_pmodel.get_I_at_time(lambda_dr=lam_prior, t_cur=t),
                              dtype=float).flatten() for t in times]
        return times, samples


    def _extract_measurements(state_dict: Dict, measurement_key: str, scale_sigma_by_mean: bool = False):
        aggregated: Dict[float, Tuple[float, float]] = {}
        for t, sublevels in state_dict.items():
            for state in sublevels.values():
                measurement = state.get(measurement_key, {})
                mean = measurement.get("mean")
                sigma = measurement.get("sigma")
                if mean is None or sigma is None:
                    continue
                time_value = measurement.get("time", t)
                means = np.atleast_1d(mean).astype(float)
                sigmas = np.atleast_1d(sigma).astype(float)
                times = np.atleast_1d(time_value).astype(float)

                if times.size == 1 and means.size > 1:
                    times = np.full_like(means, times.item(), dtype=float)
                if sigmas.size == 1 and means.size > 1:
                    sigmas = np.full_like(means, sigmas.item(), dtype=float)

                for tm, mu, sig in zip(times, means, sigmas):
                    adjusted_sigma = sig * mu if scale_sigma_by_mean else sig
                    aggregated[float(tm)] = (float(mu), float(adjusted_sigma))

        times_sorted = np.array(sorted(aggregated.keys()), dtype=float)
        if times_sorted.size == 0:
            return times_sorted, np.array([], dtype=float), np.array([], dtype=float)
        means_sorted = np.array([aggregated[t][0] for t in times_sorted], dtype=float)
        sigmas_sorted = np.array([aggregated[t][1] for t in times_sorted], dtype=float)
        return times_sorted, means_sorted, sigmas_sorted

    series_by_method = {label: _collect_series(pdt) for label, pdt in methods}

    PRIOR_COLOR = (0.55, 0.55, 0.55)
    for show_prior in prior_variants:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        for (label, pdt), color in zip(methods, colors):
            summary = series_by_method[label]
            legend_label = "Updated Distribution"

            _plot_series_violin(axs[0], summary["lambda"]["times"], summary["lambda"]["samples"], color, legend_label)
            _plot_series_violin(axs[1], summary["I"]["times"], summary["I"]["samples"], color, legend_label)

        if show_prior:
            prior_times, prior_samples = _prior_I_series(pdts[0])
            _plot_series_violin(axs[1], prior_times, prior_samples, PRIOR_COLOR, "Prior Distribution", side="left")

        # Observations
        prop_times, prop_means, prop_sigmas = _extract_measurements(pdts[0].state_dict, "property_measurement")
        if prop_times.size > 0:
            axs[0].errorbar(
                prop_times, prop_means, yerr=prop_sigmas, fmt="x", color=MEAS_COLOR, label="Observation", capsize=4
            )

        # True values from the physical twin.
        timestep_set = set(int(float(t)) for t in np.atleast_1d(timeframe))
        timestep_set.add(0)
        timesteps_cont = np.array(sorted(timestep_set), dtype=int)

        true_lambda = np.repeat(float(physical_twin.true_lambda), len(timesteps_cont))
        axs[0].plot(timesteps_cont, true_lambda, linestyle="--", color="black", label="True Value", zorder=10)

        times_to_extract = sorted(list(set(list(measured_corrosion_times) + [0])))
        timesteps_measured = np.array(times_to_extract, dtype=int)

        true_deterioration = physical_twin.displacement_pmodel.get_loss_at_time(
            t=timesteps_measured, lambda_dr=np.repeat(float(physical_twin.true_lambda), len(timesteps_measured))
        )
        true_I = physical_twin.displacement_pmodel.get_I_from_loss(loss=true_deterioration)
        true_I = np.array(true_I, dtype=float).reshape(-1)

        axs[1].scatter(timesteps_measured, true_I, color="black", label="True Value", marker="x", zorder=10)

        # Axis formatting
        axs[0].set_xlabel("Time [years]")
        axs[0].set_ylabel("Deterioration Rate [%/year]")
        axs[0].set_xlim(left=-0.5)

        axs[1].set_xlabel("Time [years]")
        axs[1].set_ylabel("Moment of Inertia [$m^4$]")
        axs[1].set_xlim(left=-0.5)

        axs[0].xaxis.set_major_locator(plt.MultipleLocator(1))
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(1))

        # Single legend below figure
        legend_below(fig)

        # Deduce output directory.
        if plot_dir is None:
            base_dir = getattr(pdts[0], "plot_dir", None)
            if base_dir:
                plot_dir = Path(base_dir).parent / "comparison"
            else:
                plot_dir = Path("plots") / "comparison"
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        names_str = "_".join(methods_names)
        suffix = "_with_prior" if show_prior else ""
        filename = plot_dir / f"scenario_comparison_property_{names_str}_{time_id}_violin{suffix}.png"

        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
