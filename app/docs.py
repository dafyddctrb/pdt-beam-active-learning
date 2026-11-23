"""Companion notes for the cantilever-beam app.

TOPICS supplies the introduction, section notes, and references. PARAMS links
control tooltips to those notes, and HOME records each topic's location.
"""

TOPICS = {
    "overview": dict(
        title="What this tool does",
        status="ok",
        long=r"""
This app accompanies the cantilever-beam example in Section 5 of *Bayesian Data
Integration and Model Updating in Probabilistic Digital Twins* (see References).

Deterioration readings inform the deterioration rate $\lambda$. Displacement
measurements depend on Young's modulus $E$ and the second moment of area $I(t)$,
which is determined by $\lambda$ and the beam geometry. Together, these data update
uncertainty in $E$ and $\lambda$.

The app runs sequential and batch updating with either the physics-based model
or a Gaussian-process surrogate refined by active learning. The analytical
beam response and two unknowns make a numerical reference posterior feasible.
""",
    ),
    "scope": dict(
        title="Paper example and optional settings",
        status="ok",
        long=r"""
In section 1, load **(defaults)** under **Configuration file** for the paper's
20-year cantilever setup. It uses annual displacement observations and
deterioration observations in years 5, 10 and 15. Use **Save as** and **Save**
to keep your settings as a JSON file in §configs/§.

The simply supported beam, alternative measurement settings, and additional
active-learning criteria let you explore variations of the paper example.
The app and command-line scripts share the settings in §case_study_config.py§.
""",
    ),
    "model": dict(
        title="Beam model",
        status="ok",
        long=r"""
**Structural system** sets the support and load type. For the cantilever, enter
a distributed load $q$ in kN/m. For the simply supported beam, enter a midspan
point load $V$ in kN. Set the span $l$, width $b$, and height $h$ in metres.

The analytical maximum deflections are

$$w = \frac{q l^4}{8 E I(t)} \quad \text{(cantilever)},$$

$$w = \frac{V l^3}{48 E I(t)} \quad \text{(simply supported)}.$$

**True E** in GPa and **True λ** in % per year set the synthetic physical twin.
Each year the beam loses $\lambda$ percent of its remaining second moment of area.

$$I(t) = I_0 (1-\lambda/100)^t,\qquad I_0 = \frac{b h^3}{12}.$$

The same true values also scale the prior means configured below.
""",
    ),
    "measurements": dict(
        title="Synthetic measurements and likelihood",
        status="ok",
        long=r"""
**First year** and **Last year** set the time frame. Displacement observations
start in the first year. Deterioration observations have their own first and
last year. The two interval controls set the spacing in years. The schedule
plot shows the years actually selected.

- **Displacement noise (relative)** sets the Gaussian noise standard deviation
  as a fraction of the true displacement. For example, 0.01 means 1%.
- **Deterioration noise (relative)** sets the Gaussian noise standard deviation
  as a fraction of the true reduction $I_0 - I(t)$. For example, 0.10 means 10%
  of that reduction. The noisy moment of inertia $z_{I,t}$ is converted to a
  reading of $\lambda$ with the inverse deterioration law at the measurement year,
  $z_{\lambda,t} = 100\,[1-(z_{I,t}/I_0)^{1/t}]$.
- **Likelihood σ of displacement** and **Likelihood σ of λ** set the assumed
  Gaussian error standard deviations used for updating, in metres and % per
  year, respectively. The error bars in section 2 show these likelihood settings.
  The two relative noise controls set the scatter of the generated data.

In section 2, **Measurement seed** fixes the random draws for a given configuration.
Click **Generate measurements** to apply the seed and settings. Generate the
measurements again after changing the beam, true values, schedule, noise, or
likelihood standard deviations. The generated data are shared by the methods
in a single-data-set run or an AL comparison.
""",
    ),
    "prior": dict(
        title="Prior of the digital twin",
        status="ok",
        long=r"""
The twin starts with independent Normal priors for $E$ and $\lambda$.

- **Prior mean / true value** sets the centre of each prior. For example, 0.85
  places the prior mean at 85% of the corresponding true value.
- **Prior CoV** sets the standard deviation as a fraction of the prior mean.
  For example, 0.15 gives a standard deviation of 15% of that mean.

The prior plot updates as you adjust these controls. The star marks the true values.
""",
    ),
    "updating": dict(
        title="Sequential and batch updating",
        status="ok",
        long=r"""
At a deterioration inspection, numerical integration updates $\lambda$ from its
original prior using all deterioration observations to date (Eq. 6 of Paper). Displacement
observations then update $E$ and $\lambda$ jointly through aBUS.

- **Seq** (Eq. 8 of Paper) incorporates the new displacement using independent Normal priors fitted
  to the preceding posterior samples. When a deterioration reading arrives,
  the update starts from the original $E$ prior and the deterioration-only
  $\lambda$ posterior, then replays the displacement history in order.
- **Batch** (Eq. 7 of paper) starts each displacement update from that same property-only state
  and uses all displacements available at that time together.

**PM** uses the physics-based beam model. **SRG** uses the Gaussian-process
surrogate with active learning. In **Single data set** mode, select the methods
to run on the observations from section 2. **Run seed** controls the sampling
and surrogate initialization for this run.

In **Repeated study**, set **Runs** and **Base seed**. All four methods use the same
new data set within each repetition. The first repetition uses the base seed
for data generation and sampling, and each subsequent repetition increases it by one.

In **AL comparison**, choose **Strategies**, **Learning functions**, and
**Stopping criteria**. Every selected combination uses the same observations.
**Parallel workers** limits the number of combinations run concurrently.
This mode uses the per-criterion thresholds in §run_active_learning_comparison.py§.

In single and repeated modes, **Compute the reference posterior afterwards**
runs an additional grid and Monte Carlo analysis to produce the accuracy table.
""",
    ),
    "abus": dict(
        title="aBUS with Subset Simulation",
        status="ok",
        long=r"""
Adaptive Bayesian Updating with Structural reliability methods (**aBUS**) samples
the posterior through the event $p\le cL(\mathbf{x})$, where $p$ is an auxiliary
uniform variable, $L$ the likelihood, and $c$ the scaling factor adapted during
sampling. **Subset Simulation** reaches this event through intermediate levels.

- **Samples per level, physics model** and **Samples per level, surrogate**
  set the population sizes separately for PM and SRG. Each batch likelihood
  evaluation uses the response at all displacement times included in that update.
- **SuS conditional probability p0** is the target fraction of samples retained
  as seeds at an intermediate level. For example, 0.10 retains about a tenth of
  the population and uses those seeds to sample the next level.

See the paper and the method references for details.
""",
    ),
    "surrogate": dict(
        title="Gaussian-process surrogate",
        status="ok",
        long=r"""
The surrogate predicts displacement from $(E,I)$. Time enters through $I(t)$, so
a surrogate for a fixed beam and load can be reused as the twin is updated.

The Gaussian process starts with five physics-model evaluations in this
two-parameter example. It uses a constant mean and an RBF kernel on standardized
data. Its predictive mean supplies the displacement estimate, and its predictive
standard deviation is used by active learning.

**GP training epochs** sets the number of optimizer steps in each active-learning
refit. **GP learning rate** sets the Adam step size for those refits. The initial
fit uses the GP training routine's defaults.
""",
    ),
    "al": dict(
        title="Active learning with the U-function",
        status="ok",
        long=r"""
The **Learning function** control selects how surrogate uncertainty is propagated
to the log-likelihood. **First order** uses the local slope of the log-likelihood.
**Second order** also includes curvature terms in its mean and variance estimates.

At a Subset Simulation level with threshold $b_k$, the U-function measures the
distance from the estimated BUS limit state to the threshold relative to its
uncertainty.

$$U = \frac{|\mu_{\hat h}-b_k|}{\sigma_{\hat h}}.$$

Active learning evaluates the candidate with the smallest $U$ using the physics
model and refits the surrogate with that point.

**Max AL iterations per update** caps the number of points added in one enrichment
call at a Subset Simulation level. **Minimum training-set size** counts all points
already collected, including the initial design. The stopping criterion can
declare convergence once this size is reached. The iteration cap can end the
call earlier.
""",
    ),
    "stopping": dict(
        title="Stopping criteria",
        status="ok",
        long=r"""
**Stopping criterion** sets the check for ending the current enrichment call.
In these controls, $P_f$ denotes the
estimated fraction of candidate samples in the auxiliary BUS event.

- **Learning-function based (min U):** checks whether all candidates clear
  **Threshold on min U**. The default threshold is 2.
- **Stability of Pf:** checks the relative change in $P_f$ between successive
  estimates. This is the paper setting. It requires at least one additional
  training point per enrichment call.
- **Bounds on Pf:** checks the relative gap between the estimates formed from
  the lower and upper surrogate bounds.
- **Stability of β / Bounds on β:** apply the corresponding check to the
  transformed index $\beta=-\Phi^{-1}(P_f)$. **Combined** requires both β
  checks to pass twice in a row.
- **Re-anchored change of the failure fraction:** holds the previous BUS level
  fixed and checks the absolute change in the fraction below that level.

**Convergence tolerance (relative change of Pf)** is used by the stability and
bounds criteria, with conversion to β units where needed. A value of 0.05 means
a 5% relative tolerance. The re-anchored criterion uses **Tolerance ε** in
absolute fraction units. A value of 0.05 there means five percentage points.
""",
    ),
    "reference": dict(
        title="Numerical reference posterior",
        status="ok",
        long=r"""
The reference posterior is computed by grid integration over $(E,\lambda)$.
It uses the physics-based beam model and the same priors, observations, and
assumed measurement errors as the updating methods. The optional reference
analysis also computes a Monte Carlo cross-check by weighting prior samples
by their likelihood.

- **Run** opens a saved run. **Show the configuration of this run** displays
  the settings used to create it. For a repeated study, **Show** opens the
  selected repetition, and **Show as LaTeX rows** formats the summary table.
- **Time step t** changes the year shown in the posterior and sample views.
  The reference at each year uses the observations available by that year.
- **Distance to the reference posterior** selects a final-step measure.
  Wasserstein-1 compares one marginal at a time. Energy distance compares
  the joint $(E,\lambda)$ samples.
- In an AL comparison, **Error measure** selects the held-out region used for
  the normalized surrogate error. **Show all metrics of this run** opens the
  saved comparison table.

The app's true-model call counts include physics-based evaluations across
the update history. For surrogate methods, these are the initial training
evaluations and the points added by active learning.
""",
    ),
    "workflow": dict(
        title="How to use this page",
        status="ok",
        long=r"""
1. **Numerical investigation.** Load the defaults or set up a configuration.
2. **Synthetic measurements.** Generate a data set for single and AL comparison runs.
3. **Run.** Select a mode and start the calculation. Runs continue in the background.
4. **Results.** Open a saved run. App runs and their settings are stored in §app_runs/§
   by default, and supported command-line runs from §results/§ appear in the same list.

The README provides installation and run commands. Use §run_method_comparison.py§
for a scripted method comparison or §run_active_learning_comparison.py§ for an
active-learning comparison.
""",
    ),
    "references": dict(
        title="References",
        status="draft",
        long="""
**This work**

- Cotoarbă, D., Mavritsakis, A., Schweckendiek, T., Smith, I. F. C., Straub, D. Bayesian
  data integration and model updating in probabilistic digital twins. *Manuscript in
  preparation* (venue, year and DOI to be added). This tool reproduces its Section 5.
- Cotoarbă, D., Straub, D., Smith, I. F. C. (2025). Probabilistic digital twins for
  geotechnical design and construction. *Data-Centric Engineering* 6, e30.
  doi:10.1017/dce.2025.10008.

**Methods**

- Straub, D., Papaioannou, I. (2015). Bayesian updating with structural reliability
  methods. *Journal of Engineering Mechanics* 141, 04014134.
  doi:10.1061/(ASCE)EM.1943-7889.0000839.
- Betz, W., Papaioannou, I., Beck, J. L., Straub, D. (2018). Bayesian inference with
  subset simulation: strategies and improvements. *Computer Methods in Applied Mechanics
  and Engineering* 331, 72–93. doi:10.1016/j.cma.2017.11.021.
- Au, S. K., Beck, J. L. (2001). Estimation of small failure probabilities in high
  dimensions by subset simulation. *Probabilistic Engineering Mechanics* 16, 263–277.
  doi:10.1016/S0266-8920(01)00019-4.
- Echard, B., Gayton, N., Lemaire, M. (2011). AK-MCS: an active learning reliability
  method combining Kriging and Monte Carlo simulation. *Structural Safety* 33, 145–154.
  doi:10.1016/j.strusafe.2011.01.002.
- Moustapha, M., Marelli, S., Sudret, B. (2022). Active learning for structural
  reliability: survey, general framework and benchmark. *Structural Safety* 96, 102174.
  doi:10.1016/j.strusafe.2021.102174.

*Draft: add the venue, year and DOI of this work once known.*
""",
    ),
}

# Short control descriptions with pointers to the corresponding section notes.
PARAMS = {
    "beam_type": ("Choose the cantilever with a distributed load or the simply supported beam with a midspan point load.",
                  "model"),
    "l": ("Beam span [m].", "model"),
    "V": ("Distributed load [kN/m] for the cantilever or midspan point load [kN] for the simply supported beam.",
          "model"),
    "b": ("Rectangular cross-section width [m].", "model"),
    "h": ("Rectangular cross-section height [m].", "model"),
    "true_E": ("Young's modulus [GPa] used to generate the synthetic data and scale the prior mean.", "model"),
    "true_lambda": ("Annual loss of the remaining second moment of area [% per year] in the physical twin.", "model"),
    "prior_E_factor": ("Prior mean of E divided by its synthetic true value.", "prior"),
    "prior_E_cov": ("Prior standard deviation of E divided by its prior mean.", "prior"),
    "prior_lambda_factor": ("Prior mean of λ divided by its synthetic true value.", "prior"),
    "prior_lambda_cov": ("Prior standard deviation of λ divided by its prior mean.", "prior"),
    "t_start": ("First year of the simulation and first displacement measurement.", "measurements"),
    "t_end": ("Final simulation year. The schedule plot shows which measurements occur before or at this year.",
              "measurements"),
    "t_step_displacement": ("Years between displacement measurements.", "measurements"),
    "t_start_deterioration": ("Year of the first scheduled deterioration measurement.", "measurements"),
    "t_step_deterioration": ("Years between deterioration measurements.", "measurements"),
    "t_end_deterioration": ("Last year in which a deterioration measurement may be taken (capped at the last year).", "measurements"),
    "displacement_noise_rel": ("Synthetic noise standard deviation as a fraction of the true displacement.",
                               "measurements"),
    "displacement_sigma": ("Assumed displacement-error standard deviation in the likelihood [m].", "measurements"),
    "loss_noise_rel": ("Synthetic noise standard deviation as a fraction of the true reduction I₀ − I(t), before converting to λ.",
                       "measurements"),
    "sigma_lambda": ("Assumed standard deviation in the likelihood for the converted λ readings [%/year].",
                     "measurements"),
    "n_samples_pm": ("Population size at each Subset Simulation level for the physics-based methods.", "abus"),
    "n_samples_srg": ("Population size at each Subset Simulation level for the surrogate-based methods.", "abus"),
    "p0": ("Target fraction of samples retained as seeds at an intermediate Subset Simulation level.", "abus"),
    "acquisition_function": ("Order of the log-likelihood uncertainty approximation used in the U-function.", "al"),
    "convergence_method": ("Rule for stopping active learning at a Subset Simulation level.", "stopping"),
    "convergence_criteria": ("Relative Pf tolerance for stability and bounds, converted for the β criteria.", "stopping"),
    "min_u_threshold": ("Minimum U-value every candidate must reach under the min-U rule.", "stopping"),
    "seeded_delta_eps": ("Absolute tolerance on the change in fraction at the frozen BUS level.", "stopping"),
    "n_iterations": ("Maximum points added in one enrichment call at a Subset Simulation level.", "al"),
    "min_train_points": ("Minimum total number of training points for the stopping criterion to declare convergence.",
                         "al"),
    "epochs": ("Number of Adam optimization steps per active-learning refit.", "surrogate"),
    "lr": ("Adam step size for active-learning refits.", "surrogate"),
}

# "0" is the introduction, numbered sections have side notes, and "refs" is the references section.
HOME = {"overview": "0", "scope": "0", "workflow": "0", "references": "refs",
        "model": "1", "prior": "1", "measurements": "1", "abus": "1", "surrogate": "1", "al": "1",
        "stopping": "1", "updating": "3", "reference": "4"}
GENERAL = [key for key in TOPICS if HOME[key] == "0"]
assert set(HOME) == set(TOPICS)
assert all(topic in TOPICS for _, topic in PARAMS.values())


def where(topic: str) -> str:
    """Return the app location of a topic."""
    title = TOPICS[topic]["title"]
    if HOME[topic] == "0":
        return f"More in the introduction under {title}."
    if HOME[topic] == "refs":
        return "See References at the end of the page."
    return f"See {title} in the notes beside section {HOME[topic]}."


def tip(param: str) -> str | None:
    """Return a control explanation and its topic pointer, if known."""
    if param not in PARAMS:
        return None
    text, topic = PARAMS[param]
    return f"{text}\n\n{where(topic)}"


def topic_tip(topic: str) -> str:
    """Return a topic pointer for help text outside the configuration controls."""
    return where(topic)
