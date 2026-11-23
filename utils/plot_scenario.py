import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime

from utils.plot_style import set_academic_style, legend_below, PRED_COLOR, TRUE_COLOR, MEAS_COLOR, PRIOR_COLOR

def plot_scenario(PDT, PHT, timeframe: np.ndarray, title: str, plot_dir: str = "plots"):
    """
    Plots the evolution of the Physical Digital Twin (PDT) state against the Physical Twin (PHT) true values.
    """
    sns.set_theme(style="whitegrid")
    set_academic_style()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    prediction_color = PRED_COLOR
    true_color = TRUE_COLOR

    t_min, t_max = timeframe.min(), timeframe.max()
    if t_max == t_min:
        get_alpha = lambda t: 1.0
    else:
        get_alpha = lambda t: 0.3 + 0.7 * (t - t_min) / (t_max - t_min)

    observation_labeled = False
    for i, t_cur in enumerate(timeframe):
        sublevels = PDT.state_dict[t_cur].keys()
        # Use the latest sublevel for predicted values
        sublevel = max(sublevels)
        alpha = get_alpha(t_cur)
        label_predicted = 'Predicted' if i == len(timeframe) - 1 else None

        # Corrosion Rate
        axs[0, 0].errorbar(
            t_cur,
            PDT.state_dict[t_cur][sublevel]['lambda']['mu'],
            yerr=PDT.state_dict[t_cur][sublevel]['lambda']['sigma'],
            fmt='o',
            color=prediction_color,
            alpha=alpha,
            label=label_predicted
        )

        # Moment of Inertia
        axs[1,1].errorbar(
            t_cur,
            PDT.state_dict[t_cur][sublevel]['I']['mu'],
            yerr=PDT.state_dict[t_cur][sublevel]['I']['sigma'],
            fmt='o',
            color=prediction_color,
            alpha=alpha,
            label=label_predicted
        )

        # Displacement
        axs[0, 1].errorbar(
            t_cur,
            PDT.state_dict[t_cur][sublevel]['displacement']['mu'],
            yerr=PDT.state_dict[t_cur][sublevel]['displacement']['sigma'],
            fmt='o',
            color=prediction_color,
            alpha=alpha,
            label=label_predicted
        )

        # Find measurements: property may be in an earlier sublevel, behavior in the latest.
        # Only plot one measurement per type per timestep.
        prop_meas = None
        beh_meas = None
        for sub in sorted(sublevels):
            state = PDT.state_dict[t_cur][sub]
            if state['property_measurement']['mean'] is not None:
                prop_meas = state['property_measurement']
            if state['behavior_measurement']['mean'] is not None:
                beh_meas = state['behavior_measurement']

        # Property measurement (deterioration rate / lambda)
        if prop_meas is not None:
            label_obs = 'Observation' if not observation_labeled else None
            if isinstance(prop_meas['mean'], np.ndarray):
                matches = np.where(np.atleast_1d(prop_meas['time']) == t_cur)[0]
                if len(matches) > 0:
                    cur_i = matches[0]
                    axs[0, 0].errorbar(
                        t_cur, prop_meas['mean'][cur_i],
                        yerr=prop_meas['sigma'][cur_i],
                        fmt='x', color=MEAS_COLOR, alpha=alpha, label=label_obs
                    )
                    observation_labeled = True
            else:
                axs[0, 0].errorbar(
                    t_cur, prop_meas['mean'],
                    yerr=prop_meas['sigma'],
                    fmt='x', color=MEAS_COLOR, alpha=alpha, label=label_obs
                )
                observation_labeled = True

        # Behavior measurement (displacement) — sigma is stored as absolute std
        if beh_meas is not None:
            label_obs = 'Observation' if not observation_labeled else None
            if isinstance(beh_meas['mean'], np.ndarray):
                matches = np.where(np.atleast_1d(beh_meas['time']) == t_cur)[0]
                if len(matches) > 0:
                    cur_i = matches[0]
                    axs[0, 1].errorbar(
                        t_cur, beh_meas['mean'][cur_i],
                        yerr=beh_meas['sigma'][cur_i],
                        fmt='x', color=MEAS_COLOR, alpha=alpha, label=label_obs
                    )
                    observation_labeled = True
            else:
                axs[0, 1].errorbar(
                    t_cur, beh_meas['mean'],
                    yerr=beh_meas['sigma'],
                    fmt='x', color=MEAS_COLOR, alpha=alpha, label=label_obs
                )
                observation_labeled = True

    # Young's Modulus: prior (first timestep) vs posterior (last timestep)
    first_t = timeframe[0]
    first_sub = sorted(PDT.state_dict[first_t].keys())[0]
    E_prior = PDT.state_dict[first_t][first_sub]['E']['samples']

    last_t = timeframe[-1]
    last_sub = sorted(PDT.state_dict[last_t].keys())[-1]
    E_posterior = PDT.state_dict[last_t][last_sub]['E']['samples']

    sns.histplot(E_prior, bins=20, ax=axs[1, 0], color=PRIOR_COLOR, alpha=0.5,
                 kde=True, stat="density", linewidth=0, label='Prior')
    sns.histplot(E_posterior, bins=20, ax=axs[1, 0], color=prediction_color, alpha=0.6,
                 kde=True, stat="density", linewidth=0, label='Posterior')

    # Plot true values derived from the physical twin
    true_lambda = np.repeat(float(PHT.true_lambda), len(timeframe))
    true_E = np.repeat(float(PHT.true_E), len(timeframe))
    true_deterioration = PHT.displacement_pmodel.get_loss_at_time(
        t=timeframe, lambda_dr=true_lambda
    )
    true_I = PHT.displacement_pmodel.get_I_from_loss(loss=true_deterioration)
    true_displacement = PHT.displacement_pmodel.get_beam_max_deflection(E=true_E, I=true_I)
    true_displacement = np.array(true_displacement, dtype=float).reshape(-1)

    axs[0, 0].plot(timeframe, true_lambda, label='True Value', color=true_color, linestyle='--')
    axs[1, 0].axvline(true_E[0], label='True Value', color=true_color, linestyle='--')
    axs[1, 1].plot(timeframe, np.array(true_I, dtype=float).reshape(-1), label='True Value', color=true_color, linestyle='--')
    axs[0, 1].plot(timeframe, true_displacement, label='True Value', color=true_color, linestyle='--')

    # --- Set titles and labels ---
    axs[0, 0].set_xlabel('Time [years]')
    axs[0, 0].set_ylabel('Deterioration Rate [1/year]')

    axs[0, 1].set_xlabel('Time [years]')
    axs[0, 1].set_ylabel('Displacement [m]')

    axs[1, 0].set_xlabel('Young\'s Modulus [$N/m^2$]')
    axs[1, 0].set_ylabel('Probability Density')

    axs[1, 1].set_xlabel('Time [years]')
    axs[1, 1].set_ylabel('Moment of Inertia [$m^4$]')

    # Single legend below figure
    legend_below(fig)

    # save figure
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"{plot_dir}/{title}_{time_id}.png", bbox_inches='tight')
    plt.close()
