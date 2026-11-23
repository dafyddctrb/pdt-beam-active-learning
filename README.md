# Probabilistic Digital Twin of a Deteriorating Cantilever Beam

This repository accompanies the cantilever-beam case study (Section 5) in
*Bayesian Data Integration and Model Updating in Probabilistic Digital Twins*.
It uses synthetic deterioration and displacement measurements to compare sequential
and batch Bayesian updating, each with either a physics-based model or a
Gaussian-process (GP) surrogate refined through active learning. The four combinations are
called **Seq PM**, **Seq SRG**, **Batch PM** and **Batch SRG** (PM: physics-based model,
SRG: surrogate), and each is scored against a numerical reference posterior.

## Installation

```bash
git clone https://github.com/dafyddctrb/pdt-beam-active-learning.git
cd pdt-beam-active-learning
conda env create -f environment.yml
conda activate pdt-beam-al
```

or `pip install -r requirements.txt` with Python 3.10.

## Getting started: the web app

Launch the local browser app from the repository root.

```bash
streamlit run app/app.py
```

1. In **Numerical investigation**, load **(defaults)** under **Configuration file**.
2. In **Synthetic measurements**, click **Generate measurements**.
3. In **Run**, choose **Single data set** and click **Run 4 methods**.
4. When the run finishes, open it in **Results** and use the time slider to explore
   the posterior at different years.

https://github.com/user-attachments/assets/3f77ec59-a47a-4e19-9ba3-d44297d1aacf

The video follows a single cantilever example from configuration to results. In the
results view, the time slider shows how the posterior distributions evolve as new
observations become available.

The app runs entirely on your machine. You can change the beam, measurement,
and updating settings, or run a repeated study or an active-learning comparison.
The notes beside each section explain the available controls.

Runs are stored in `app_runs/` (one folder per run, with its `config.json`) and can be
reopened later. Studies run from the command line (in `results/`) appear in the same
run list.

The app calls the same code as the scripts. A **Repeated study** with base seed *S* gives
the same numbers as `run_method_comparison.py --seed S` with the same config. In **Single
data set** mode the measurements and the run have separate seeds, so its numbers are
reproducible within the app but differ from a command-line run.

The repository is self-contained. An AI coding assistant such as Claude Code or Codex
can also install the environment and launch the app.

## Command line

Everything the app does can also be run with scripts: `run_method_comparison.py` (the four
methods on one or more synthetic data sets) and `run_active_learning_comparison.py` (learning
functions × stopping criteria). Settings come from a JSON file (`--config`, see `configs/`) or
one flag per setting. `--help` lists them all.

## Repository layout

```
case_study_config.py                 all settings of the numerical investigation (CaseStudyConfig)
run_method_comparison.py             Section 5: Seq/Batch x PM/SRG on N synthetic data sets
analyse_method_comparison.py         Section 5.2.3: scores against the reference posterior, final-posterior figure
reference_posterior.py               numerical reference posterior (grid integration + Monte Carlo)
run_active_learning_comparison.py    optional: learning functions x stopping criteria
twins/                               digital twin (priors, surrogate, updating hooks) and physical twin
                                     (synthetic ground truth and measurements)
behavior_models/                     beam mechanics (beams.py), GP surrogate and active learning
                                     (gp_surrogate.py)
updating/                            aBUS-SuS with active learning, aCS sampler, likelihood,
                                     property updating, ERA distributions
utils/                               plotting helpers, true-model call counting
app/                                 web app
configs/                             example configs
```

## Project Information
- **Authors:** Dafydd Cotoarbă, Antonis Mavritsakis, Timo Schweckendiek, Ian F. C. Smith, Daniel Straub
- **Organization:** ERA Group and TUM Georg Nemetschek Institute, TUM
- **Publication:** Cotoarbă, D., Mavritsakis, A., Schweckendiek, T., Smith, I. F. C., & Straub, D. Bayesian Data
  Integration and Model Updating in Probabilistic Digital Twins. Manuscript in preparation.

---

## Citation
If you find it useful for your research, please consider citing it:

**BibTeX Entry:**
```bibtex
@unpublished{cotoarba2026bayesian,
  author = {Cotoarb{\u{a}}, Dafydd and Mavritsakis, Antonis and Schweckendiek, Timo and
            Smith, Ian F. C. and Straub, Daniel},
  title  = {Bayesian Data Integration and Model Updating in Probabilistic Digital Twins},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## License and acknowledgements

MIT License, see `LICENSE`. Parts of `updating/` are adapted from the software of
the Engineering Risk Analysis Group, TU Munich. See `NOTICE.md` for details.
