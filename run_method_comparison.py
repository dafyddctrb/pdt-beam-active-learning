"""Numerical investigation of the paper, Section 5 (cantilever beam): comparison of the four
updating methods.

A vertical steel cantilever deteriorates over the years (its second moment of area I decays at
the rate lambda). Property data (deterioration readings) update lambda directly, by numerical
integration (Section 3.3); behaviour data (displacements) update (E, lambda) with aBUS-SuS
(Section 3.4). The behaviour update runs in four ways:

    Seq PM     sequential updating, physics-based model
    Seq SRG    sequential updating, GP surrogate refined by active learning (Section 4)
    Batch PM   batch updating (all data so far, at every step), physics-based model
    Batch SRG  batch updating, GP surrogate with active learning

Each run draws one synthetic data set from the physical twin and updates all four methods on it;
`--n-runs` repeats this with new data. Afterwards analyse_method_comparison.py scores every method
against the exact posterior (Section 5.2.3: Table 2 and the figure of the final posteriors over all
runs). The settings (Section 5.1, Table 1) come from case_study_config.py, a --config JSON or flags.

    python run_method_comparison.py --config configs/paper_section5.json --n-runs 50 --seed 1000 \
        --results-folder results/sigma_d_0p02
"""
import numpy as np
from updating.AL_aBUS_SuS import AL_aBUS_SuS
from updating.property_updating import update_normal_by_integration
from utils.plot_scenario import plot_scenario
from utils.plot_style import plots_enabled
from utils.plot_scenario_comparison import plot_scenario_comparison, plot_scenario_comparison_property
from analyse_method_comparison import main as analyse_method_comparison
from case_study_config import CaseStudyConfig, add_config_cli_args, config_from_cli
from datetime import datetime
import glob
import json
import os
import copy


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def update_displacement(PDT, PT, model_type: str = "pmodel", active_learning: bool = False, N=1000, p0=0.1, al_kwargs: dict = None):
    if model_type == "pmodel":
        samplesX, displacements = AL_aBUS_SuS(N=N, p0=p0, l_class=PDT.displacement_l,
                                                    srg_prediction_model=PDT.pmodel_predict_displacement,
                                                    active_learning=False,
                                                    retrain_srg_model=PDT.srg_retrain_model,
                                                    distr=PDT.joint_distribution,
                                                    al_kwargs=al_kwargs)
    elif model_type == "srg" and active_learning:
        samplesX, displacements = AL_aBUS_SuS(N=N, p0=p0, l_class=PDT.displacement_l,
                                                    srg_prediction_model=PDT.srg_predict_displacement,
                                                    active_learning=True,
                                                    retrain_srg_model=PDT.srg_retrain_model,
                                                    distr=PDT.joint_distribution,
                                                    al_kwargs=al_kwargs)
    elif model_type == "srg" and not active_learning:
        samplesX, displacements = AL_aBUS_SuS(N=N, p0=p0, l_class=PDT.displacement_l,
                                                    srg_prediction_model=PDT.srg_predict_displacement,
                                                    active_learning=False,
                                                    retrain_srg_model=PDT.srg_retrain_model,
                                                    distr=PDT.joint_distribution,
                                                    al_kwargs=al_kwargs)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    return samplesX, displacements




def run_sequential(PDT, PT, timeframe: np.ndarray = None, results_folder: str = 'results',
                                                                model_type: str = "pmodel", active_learning: bool = False,
                                                                beam_type: str = "two_sided", displacement_measurement_dict: dict = None,
                                                                deterioration_rate_measurement_dict: dict = None,
                                                                samples_per_step: int = 100, al_kwargs: dict = None, p0: float = 0.1):

    updating_type = "CD_Sequential Updating"
    cur_title = f"{updating_type}"
    if model_type == "pmodel":
        cur_title += " with Physics Based Model"
    elif model_type == "srg" and active_learning:
        cur_title += " with SRG Prediction and Active Learning"
    elif model_type == "srg" and not active_learning:
        cur_title += " with SRG Prediction and without Active Learning"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    plt_dir = f"{results_folder}/plots/{beam_type}/{updating_type}"

    PDT.set_parameters(plot_dir=plt_dir, model_type=model_type)
    PDT.record_accuracy_snapshot(t_cur=0)   # pre-AL baseline for the accuracy trajectory
    PDT_OG = copy.deepcopy(PDT)

    measured_displacements_mus = displacement_measurement_dict["mean"]
    measured_displacements_sigmas = displacement_measurement_dict["sigma"]
    measured_displacements_times = displacement_measurement_dict["time"]
    measured_deterioration_rates_mus = deterioration_rate_measurement_dict["mean"]
    measured_deterioration_rates_sigmas = deterioration_rate_measurement_dict["sigma"]
    measured_deterioration_rates_times = deterioration_rate_measurement_dict["time"]

    
    for i_t_cur, t_cur in enumerate(timeframe):
        PDT.sublevel_timestep = 0
        performed_update = False

        if t_cur in measured_deterioration_rates_times:
            # New property data arrived — reset to prior and re-apply all updates
            trained_srg = copy.deepcopy(PDT.displacement_srg)

            # Reset distributions to prior
            PDT.lambda_distribution = copy.deepcopy(PDT_OG.lambda_distribution)
            PDT.E_distribution = copy.deepcopy(PDT_OG.E_distribution)
            PDT.joint_distribution = copy.deepcopy(PDT_OG.joint_distribution)
            PDT.displacement_srg = trained_srg

            # Apply all accumulated deterioration measurements
            i_d_tcur = np.where(measured_deterioration_rates_times == t_cur)[0][0]
            cur_measured_dict = {"measurement": {"mean": measured_deterioration_rates_mus[:i_d_tcur+1],
                                 "sigma": measured_deterioration_rates_sigmas[:i_d_tcur+1],
                                 "time": measured_deterioration_rates_times[:i_d_tcur+1]}}

            updated_deterioration_rate_distribution = update_normal_by_integration(
                prior_dist=PDT_OG.lambda_distribution,
                observations=measured_deterioration_rates_mus[:i_d_tcur+1],
                observation_stds=measured_deterioration_rates_sigmas[:i_d_tcur+1])

            PDT.update_deterioration_rate_distribution(
                new_lambda_distribution=updated_deterioration_rate_distribution,
                t_cur=t_cur, measurement_dict=cur_measured_dict)
            performed_update = True

            # Re-apply all displacement updates from start to current time on PDT directly
            if model_type == "pmodel":
                PDT.nr_likelihood_calls = 0
            for t_disp in timeframe[:i_t_cur+1]:
                if t_disp in measured_displacements_times:
                    i_d_tdisp = np.where(measured_displacements_times == t_disp)[0][0]
                    disp_measured_dict = {"measurement": {"mean": measured_displacements_mus[i_d_tdisp],
                                        "sigma": measured_displacements_sigmas[i_d_tdisp],
                                        "time": measured_displacements_times[i_d_tdisp]}}

                    PDT.cur_timestep = t_disp
                    PDT.set_measured_displacement(
                        np.array([measured_displacements_mus[i_d_tdisp]]),
                        np.array([measured_displacements_sigmas[i_d_tdisp]]),
                        np.array([t_disp]))

                    samplesX, displacements = update_displacement(
                        PDT, PT, model_type=model_type, active_learning=active_learning,
                        N=samples_per_step, p0=p0, al_kwargs=al_kwargs)
                    print("updating distributions")
                    PDT.update_distributions(
                        t_cur=t_cur, samplesX=samplesX[-1][:,0:-1],
                        displacement_samples=displacements,
                        measurement_dict=disp_measured_dict)
                    performed_update = True

            # NOTE: no counter reset/restore for "srg": the counter is cumulative
            # (initial training + AL additions persist across the replay, which uses
            # the kept surrogate). Restoring nr_calls_before here without a matching
            # reset double-counted the entire history at every corrosion timestep.
        elif t_cur in measured_displacements_times:
            PDT.cur_timestep = t_cur
            i_d_tcur = np.where(measured_displacements_times == t_cur)[0][0]
            cur_measured_dict = {"measurement": {"mean": measured_displacements_mus[i_d_tcur],
                                    "sigma": measured_displacements_sigmas[i_d_tcur],
                                    "time": measured_displacements_times[i_d_tcur]}}

            PDT.set_measured_displacement(np.array([measured_displacements_mus[i_d_tcur]]), np.array([measured_displacements_sigmas[i_d_tcur]]), np.array([t_cur]))
            samplesX, displacements = update_displacement(PDT, PT, model_type=model_type, active_learning=active_learning, N=samples_per_step, p0=p0, al_kwargs=al_kwargs)
            print("updating distributions")
            PDT.update_distributions(t_cur=t_cur, samplesX=samplesX[-1][:,0:-1], displacement_samples=displacements,
                                    measurement_dict=cur_measured_dict)
            performed_update = True

        if not performed_update:
            PDT.save_state_dict(t_cur=t_cur)

        PDT.record_accuracy_snapshot(t_cur=t_cur)   # held-out accuracy after this timestep

    if plots_enabled():
        plot_scenario(PDT, PT, timeframe, title=cur_title, plot_dir=plt_dir)

    ### Save results to json
    dict_file_name = f"state_dict_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_folder = f"{results_folder}/{beam_type}/{updating_type}/{cur_title}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    with open(os.path.join(results_folder, dict_file_name), 'w') as f:
        state_dict_to_save = {int(k): v for k, v in PDT.state_dict.items()}
        json.dump(state_dict_to_save, f, cls=NumpyArrayEncoder)
    return PDT



def run_batch(PDT_OG, PT, timeframe: np.ndarray = None, results_folder: str = 'results',
                                                                    model_type: str = "pmodel", active_learning: bool = False, beam_type: str = "two_sided",
                                                                    displacement_measurement_dict: dict = None, deterioration_rate_measurement_dict: dict = None,
                                                                    samples_per_step: int = 100, al_kwargs: dict = None, p0: float = 0.1):
    
    updating_type = "CD_Full Updating"
    cur_title = f"{updating_type}"
    if model_type == "pmodel":
        cur_title += " with Physics Based Model"
    elif model_type == "srg" and active_learning:
        cur_title += " with SRG Prediction and Active Learning"
    elif model_type == "srg" and not active_learning:
        cur_title += " with SRG Prediction and without Active Learning"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    plt_dir=f"{results_folder}/plots/{beam_type}/{updating_type}"
    
    PDT_OG.set_parameters(plot_dir=plt_dir, model_type=model_type)
    final_PDT = copy.deepcopy(PDT_OG)
    final_PDT.record_accuracy_snapshot(t_cur=0)   # pre-AL baseline for the accuracy trajectory


    measured_displacements_mus = displacement_measurement_dict["mean"]
    measured_displacements_sigmas = displacement_measurement_dict["sigma"]
    measured_displacements_times = displacement_measurement_dict["time"]
    measured_deterioration_rates_mus = deterioration_rate_measurement_dict["mean"]
    measured_deterioration_rates_sigmas = deterioration_rate_measurement_dict["sigma"]
    measured_deterioration_rates_times = deterioration_rate_measurement_dict["time"]

    # Corrosion-only lambda posterior (prior x all corrosion measurements so far).
    # Displacement-only batch steps re-anchor lambda to THIS (not to final_PDT's
    # lambda, which the displacement updates overwrite with the joint posterior).
    latest_lambda_posterior = copy.deepcopy(PDT_OG.lambda_distribution)

    for i_t_cur,t_cur in enumerate(timeframe):
        final_PDT.cur_timestep = t_cur
        final_PDT.sublevel_timestep = 0
        performed_update = False

        if t_cur in measured_deterioration_rates_times:
            cur_PDT = copy.deepcopy(PDT_OG)
            cur_PDT.cur_timestep = t_cur
            cur_PDT.displacement_srg = copy.deepcopy(final_PDT.displacement_srg)

            final_PDT.lambda_distribution = copy.deepcopy(cur_PDT.lambda_distribution)
            final_PDT.E_distribution = copy.deepcopy(cur_PDT.E_distribution)
            final_PDT.joint_distribution = copy.deepcopy(cur_PDT.joint_distribution)

            i_d_tcur = np.where(measured_deterioration_rates_times == t_cur)[0][0]

            cur_det_measured_dict = {"measurement": {"mean": measured_deterioration_rates_mus[:i_d_tcur+1], 
                                                "sigma": measured_deterioration_rates_sigmas[:i_d_tcur+1], 
                                                "time": measured_deterioration_rates_times[:i_d_tcur+1]}}

            updated_deterioration_rate_distribution = update_normal_by_integration(prior_dist=PDT_OG.lambda_distribution,
                                                                                        observations=measured_deterioration_rates_mus[:i_d_tcur+1],
                                                                                        observation_stds=measured_deterioration_rates_sigmas[:i_d_tcur+1])
            latest_lambda_posterior = copy.deepcopy(updated_deterioration_rate_distribution)

            cur_PDT.update_deterioration_rate_distribution(new_lambda_distribution=updated_deterioration_rate_distribution, t_cur=t_cur, measurement_dict=cur_det_measured_dict)
            final_PDT.update_deterioration_rate_distribution(new_lambda_distribution=updated_deterioration_rate_distribution, t_cur=t_cur, measurement_dict=cur_det_measured_dict)
            performed_update = True

            print(50*"-=-")
            print(f"Performed update for deterioration rate at time: {t_cur}")
            print(f"New sublevel: {final_PDT.sublevel_timestep}")
            print(50*"-")

            # re-apply the whole displacement history up to t_cur, also in a deterioration
            # year without a displacement reading; otherwise E would fall back to the prior
            n_disp = int(np.sum(np.asarray(measured_displacements_times) <= t_cur))
            if n_disp > 0:
                cur_PDT.nr_likelihood_calls = 0
                cur_measured_dict = {"measurement": {"mean": measured_displacements_mus[:n_disp],
                                                    "sigma": measured_displacements_sigmas[:n_disp],
                                                    "time": measured_displacements_times[:n_disp]}}

                cur_PDT.set_measured_displacement(measured_displacements_mus[:n_disp], measured_displacements_sigmas[:n_disp], measured_displacements_times[:n_disp])
                samplesX, displacements = update_displacement(cur_PDT, PT, model_type=model_type, active_learning=active_learning, N=samples_per_step, p0=p0, al_kwargs=al_kwargs)
                final_PDT.displacement_srg = copy.deepcopy(cur_PDT.displacement_srg)
                
                if model_type == "pmodel":
                    final_PDT.nr_likelihood_calls = copy.deepcopy(cur_PDT.nr_likelihood_calls)
                elif model_type == "srg":
                    final_PDT.nr_likelihood_calls += cur_PDT.nr_likelihood_calls
                else:
                    raise ValueError(f"Invalid model type: {model_type}")
                
                final_PDT.update_distributions(t_cur=t_cur, samplesX=samplesX[-1][:,0:-1], displacement_samples=displacements, 
                                                measurement_dict=cur_measured_dict)
                performed_update = True

                print(50*"-=-")
                print(f"Performed update for displacement at time: {t_cur}")
                print(f"New sublevel: {final_PDT.sublevel_timestep}")
                print(50*"-")
        elif t_cur in measured_displacements_times:
            # Proper batch update: the FULL displacement history is re-applied below,
            # so the starting distribution must contain NO displacement information —
            # E from the prior, lambda from the corrosion-only posterior (each
            # measurement enters the posterior exactly once). Running on final_PDT
            # re-applied the history on top of the previous posterior: the data was
            # double-counted every step, and the resulting tight populations kept the
            # AL certainty gate satisfied, so surrogate bias compounded uncorrected.
            cur_PDT = copy.deepcopy(PDT_OG)
            cur_PDT.cur_timestep = t_cur
            cur_PDT.displacement_srg = copy.deepcopy(final_PDT.displacement_srg)
            cur_PDT.nr_likelihood_calls = 0
            cur_PDT.update_deterioration_rate_distribution(
                new_lambda_distribution=copy.deepcopy(latest_lambda_posterior), t_cur=t_cur)

            i_di_tcur = np.where(measured_displacements_times == t_cur)[0][0]
            cur_measured_dict = {"measurement": {"mean": measured_displacements_mus[i_di_tcur],
                                    "sigma": measured_displacements_sigmas[i_di_tcur],
                                    "time": measured_displacements_times[i_di_tcur]}}

            cur_PDT.set_measured_displacement(measured_displacements_mus[:i_di_tcur+1], measured_displacements_sigmas[:i_di_tcur+1], measured_displacements_times[:i_di_tcur+1])
            samplesX, displacements = update_displacement(cur_PDT, PT, model_type=model_type, active_learning=active_learning, N=samples_per_step, p0=p0, al_kwargs=al_kwargs)
            final_PDT.displacement_srg = copy.deepcopy(cur_PDT.displacement_srg)
            if model_type == "pmodel":
                final_PDT.nr_likelihood_calls = copy.deepcopy(cur_PDT.nr_likelihood_calls)
            elif model_type == "srg":
                final_PDT.nr_likelihood_calls += cur_PDT.nr_likelihood_calls

            final_PDT.update_distributions(t_cur=t_cur, samplesX=samplesX[-1][:,0:-1], displacement_samples=displacements,
                                                measurement_dict=cur_measured_dict)
            performed_update = True
        
        if not performed_update:
            if model_type == "pmodel":
                # Batch PM stores the cost of each step's own update; no update, no calls
                final_PDT.nr_likelihood_calls = 0
            final_PDT.save_state_dict(t_cur=t_cur)

        final_PDT.record_accuracy_snapshot(t_cur=t_cur)   # held-out accuracy after this timestep

    updating_type = "CD_Full Updating"
    cur_title = f"{updating_type}"
    if model_type == "pmodel":
        cur_title += " with Physics Based Model"
    elif model_type == "srg" and active_learning:
        cur_title += " with SRG Prediction and Active Learning"
    elif model_type == "srg" and not active_learning:
        cur_title += " with SRG Prediction and without Active Learning"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    if plots_enabled():
        plot_scenario(final_PDT, PT, timeframe, title=cur_title, plot_dir=plt_dir)

    ### Save results to json
    dict_file_name = f"state_dict_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_folder = f"{results_folder}/{beam_type}/{updating_type}/{cur_title}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    with open(os.path.join(results_folder, dict_file_name), 'w') as f:
        state_dict_to_save = {int(k): v for k, v in final_PDT.state_dict.items()}
        json.dump(state_dict_to_save, f, cls=NumpyArrayEncoder)
    return final_PDT


METHODS = ("seq_pm", "seq_srg", "batch_pm", "batch_srg")
METHOD_NAMES = {"seq_pm": "Seq PM", "seq_srg": "Seq SRG", "batch_pm": "Batch PM", "batch_srg": "Batch SRG"}


def generate_measurements(cfg: CaseStudyConfig, PT=None):
    """Synthetic displacement and deterioration-rate measurements from the physical twin.
    Returns (displacement_measurement_dict, deterioration_rate_measurement_dict)."""
    PT = PT or cfg.make_physical_twin()
    disp_mus, det_mus, det_sigmas = [], [], []
    for t_cur in cfg.displacement_times:
        disp_mus.append(PT.measure_displacement(
            t_cur=np.array([t_cur]), noise_rel=cfg.displacement_noise_rel)[0][0])
    for t_cur in cfg.deterioration_times:
        mean, sigma = PT.measure_lambda(
            t_cur=t_cur, noise_rel=cfg.loss_noise_rel, sigma_lambda=cfg.sigma_lambda)
        print(f"Measured deterioration rate mean: {mean}, measured deterioration rate sigma: {sigma}")
        det_mus.append(mean)
        det_sigmas.append(sigma)
    displacement = {"mean": np.array(disp_mus), "sigma": np.full(len(disp_mus), cfg.displacement_sigma),
                    "time": np.array(cfg.displacement_times)}
    deterioration = {"mean": np.array(det_mus), "sigma": np.array(det_sigmas),
                     "time": np.array(cfg.deterioration_times)}
    return displacement, deterioration


def run_methods(cfg: CaseStudyConfig, displacement_measurement_dict: dict, deterioration_rate_measurement_dict: dict,
                results_folder: str, methods=METHODS, PT=None, on_step=None):
    """Run the selected updating methods (subset of METHODS) on one fixed measurement
    set. Returns {method: final PDTBeam}."""
    PT = PT or cfg.make_physical_twin()
    PDT = cfg.make_digital_twin()
    common = dict(PT=PT, timeframe=cfg.timeframe, results_folder=results_folder, beam_type=cfg.beam_type,
                  displacement_measurement_dict=displacement_measurement_dict,
                  deterioration_rate_measurement_dict=deterioration_rate_measurement_dict, p0=cfg.p0)
    pm = dict(model_type="pmodel", active_learning=False, samples_per_step=cfg.n_samples_pm)
    srg = dict(model_type="srg", active_learning=True, samples_per_step=cfg.n_samples_srg, al_kwargs=cfg.al_kwargs)
    out = {}
    for m in METHODS:
        if m not in methods:
            continue
        if on_step:
            on_step(METHOD_NAMES[m])
        if m.startswith("seq"):
            out[m] = run_sequential(
                PDT=copy.deepcopy(PDT), **common, **(pm if m.endswith("pm") else srg))
        else:
            out[m] = run_batch(
                PDT_OG=copy.deepcopy(PDT), **common, **(pm if m.endswith("pm") else srg))

    if plots_enabled() and len(out) == len(METHODS):
        pdts = {METHOD_NAMES[m]: out[m] for m in METHODS}
        groups = [["Seq PM", "Seq SRG"], ["Batch PM", "Batch SRG"], ["Seq PM", "Batch PM"],
                  ["Seq SRG", "Batch SRG"], ["Seq PM", "Seq SRG", "Batch PM", "Batch SRG"], ["Seq PM"], ["Seq SRG"]]
        for names in groups:
            plot_scenario_comparison(pdts=[pdts[n] for n in names], methods_names=names, physical_twin=PT,
                                     timeframe=cfg.timeframe, violin_plot=True)
            plot_scenario_comparison_property(pdts=[pdts[n] for n in names], methods_names=names, physical_twin=PT,
                                              timeframe=cfg.timeframe, measured_corrosion_times=cfg.deterioration_times)
    return out


def run_once(cfg: CaseStudyConfig = None, results_folder: str = "results/case_study"):
    """One run of the case study: synthetic measurements from the physical
    twin, then all four methods (Seq/Batch x PM/SRG) on them. Settings from `cfg`."""
    cfg = cfg or CaseStudyConfig()
    PT = cfg.make_physical_twin()
    displacement, deterioration = generate_measurements(cfg, PT)
    return run_methods(cfg, displacement, deterioration, results_folder, PT=PT)


def set_seed(seed: int):
    """Seed every RNG used by the pipeline (numpy global state and torch)."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Paper, Section 5: sequential vs batch Bayesian updating of a deteriorating "
                    "cantilever with the physics-based model (PM) and an active-learning GP surrogate (SRG). "
                    "Every setting can be set via --config JSON or its own flag.")
    p.add_argument("--results-folder", default="results/case_study")
    p.add_argument("--n-runs", type=int, default=50, help="number of independent runs")
    p.add_argument("--seed", type=int, default=None,
                   help="base random seed; run i uses seed + i (default: unseeded)")
    p.add_argument("--skip-analysis", action="store_true",
                   help="only do the runs, do not call analyse_method_comparison afterwards")
    add_config_cli_args(p)
    a = p.parse_args()
    cfg = config_from_cli(a)

    saved = os.path.join(a.results_folder, "config.json")
    if os.path.exists(saved) and CaseStudyConfig.from_json(saved) != cfg:
        raise SystemExit(f"{a.results_folder} already holds runs with a different config; "
                         f"use a new --results-folder.")
    # new runs would be pooled with the old ones by the analysis
    if glob.glob(os.path.join(a.results_folder, "**", "state_dict_run_*.json"), recursive=True):
        raise SystemExit(f"{a.results_folder} already holds runs; use a new --results-folder "
                         f"(or delete the old one) so that the analysis only sees this study.")
    cfg.save(a.results_folder)
    print(json.dumps(cfg.to_dict(), indent=2))

    for i in range(a.n_runs):
        if a.seed is not None:
            set_seed(a.seed + i)
        print(f"=== run {i + 1}/{a.n_runs}" + (f" (seed {a.seed + i})" if a.seed is not None else "") + " ===")
        run_once(cfg=cfg, results_folder=a.results_folder)
    if not a.skip_analysis:
        analyse_method_comparison(results_folder=a.results_folder, beam_type=cfg.beam_type)
