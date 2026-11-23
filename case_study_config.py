"""Single source of truth for the settings of the numerical investigation (paper, Section 5.1 and
Table 1: the cantilever beam, priors, measurement schedule and noise, sampler and surrogate).

Every experiment driver (run_method_comparison, run_active_learning_comparison) and the analysis
(analyse_method_comparison, reference_posterior) builds the beam, the priors and the
synthetic measurements from a `CaseStudyConfig`. A run writes its config to
`<results_folder>/config.json`, and the analysis reads it back, so the reference
posterior and the plotted truth always match the data the methods saw.

Defaults are the paper's settings (Table 1, sigma_delta = 0.02 m). Override them from a
JSON file (`--config path.json`, partial files are fine) and/or CLI flags.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

CONFIG_FILE_NAME = "config.json"
# Runs made before 2026-11-19 used a compounded deterioration law, D = 100((1 + lambda/100)^t - 1);
# since then the loss decays, I = I0 (1 - lambda/100)^t, and saved configs carry this key = "decay".
LEGACY_KEY = "deterioration_model"


def is_legacy_config(d: dict) -> bool:
    """True for a saved run config that used the old (compounded) deterioration law: a full
    config without the key (saved before it existed) or one that names another model."""
    if LEGACY_KEY in d:
        return d[LEGACY_KEY] != "decay"
    return "beam_type" in d and "lr" in d


@dataclass
class CaseStudyConfig:
    # --- beam geometry, load and ground truth (physical twin) -------------------
    beam_type: str = "cantilever"          # 'cantilever' | 'two_sided'
    l: float = 10.00                       # span [m]
    V: float = 4.0                         # load [kN]
    b: float = 1.00                        # section width [m]
    h: float = 0.13                        # section height [m]
    true_E: float = 200e6                  # true Young's modulus [kN/m2]
    true_lambda: float = 1.0               # true deterioration rate [%/year]

    # --- prior of the digital twin ---------------------------------------------
    prior_E_factor: float = 0.85           # prior mean of E = factor * true_E
    prior_E_cov: float = 0.15              # prior CoV of E
    prior_lambda_factor: float = 0.8       # prior mean of lambda = factor * true_lambda
    prior_lambda_cov: float = 0.2          # prior CoV of lambda

    # --- time axis and measurement schedule [years] ------------------------------
    t_start: int = 1
    t_end: int = 20
    t_step_displacement: int = 1           # displacement measured every n years
    t_start_deterioration: int = 5         # first deterioration (corrosion) measurement
    t_step_deterioration: int = 5          # deterioration measured every n years
    t_end_deterioration: int = 18          # last year a deterioration reading may be taken

    # --- measurement noise / likelihood -----------------------------------------
    displacement_noise_rel: float = 0.01   # synthetic noise on displacement, relative to true value
    displacement_sigma: float = 0.02       # likelihood std of displacement [m] (the paper's value)
    loss_noise_rel: float = 0.10           # synthetic noise on cumulative loss D, relative to true D
    sigma_lambda: float = 0.1              # likelihood std of lambda

    # --- sampler (aBUS + Subset Simulation) --------------------------------------
    n_samples_pm: int = 100                # samples per SuS level, physics-based model
    n_samples_srg: int = 1000              # samples per SuS level, surrogate model
    p0: float = 0.1                        # SuS conditional probability

    # --- active learning (passed to SRGModel.retrain_model) ----------------------
    acquisition_function: str = "bus_u_learning"   # 'bus_u_learning' | 'bus_u_learning_2nd'
    convergence_method: str = "pf_stability"       # see SRGModel.retrain_model
    convergence_criteria: float = 0.05
    min_u_threshold: float = 2.0
    seeded_delta_eps: float = 0.05
    n_iterations: int = 10                 # max AL iterations per retrain call
    min_train_points: int = 15
    epochs: int = 100                      # GP training epochs per refit
    lr: float = 1e-2                       # GP learning rate

    # ------------------------------------------------------------------------------
    @property
    def timeframe(self) -> np.ndarray:
        return np.arange(self.t_start, self.t_end + 1, 1)

    @property
    def displacement_times(self) -> np.ndarray:
        return np.arange(self.t_start, self.t_end + 1, self.t_step_displacement)

    @property
    def deterioration_times(self) -> np.ndarray:
        last = min(self.t_end_deterioration, self.t_end)
        return np.arange(self.t_start_deterioration, last + 1, self.t_step_deterioration)

    @property
    def beam_kwargs(self) -> dict:
        return dict(l=self.l, V=self.V, b=self.b, h=self.h)

    @property
    def al_kwargs(self) -> dict:
        return dict(acquisition_function=self.acquisition_function,
                    convergence_method=self.convergence_method,
                    convergence_criteria=self.convergence_criteria,
                    min_u_threshold=self.min_u_threshold,
                    seeded_delta_eps=self.seeded_delta_eps,
                    n_iterations=self.n_iterations,
                    min_train_points=self.min_train_points,
                    epochs=self.epochs, lr=self.lr)

    # --- construction helpers ----------------------------------------------------
    def make_physical_twin(self):
        from twins.physical_twin import PhysicalTwinBeam
        return PhysicalTwinBeam(true_l=self.l, true_V=self.V, timeframe=self.timeframe,
                                beam_type=self.beam_type, b=self.b, h=self.h,
                                true_E=self.true_E, true_lambda=self.true_lambda)

    def make_digital_twin(self):
        from twins.digital_twin import PDTBeam
        return PDTBeam(l=self.l, V=self.V, timeframe=self.timeframe, beam_type=self.beam_type,
                       E=self.true_E * self.prior_E_factor, E_cov=self.prior_E_cov,
                       lambda_mean=self.true_lambda * self.prior_lambda_factor,
                       lambda_cov=self.prior_lambda_cov, b=self.b, h=self.h)

    # --- (de)serialisation ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {**asdict(self), LEGACY_KEY: "decay"}        # marks the deterioration law of the run

    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, CONFIG_FILE_NAME)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "CaseStudyConfig":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k != LEGACY_KEY}   # see is_legacy_config
        d.pop("loss_noise_sigma", None)   # absolute loss noise of runs before 2026-11-20 (data generation only)
        if "t_end" in d and "t_end_deterioration" not in d:
            # configs before 2026-11-23 had a fixed cut-off two years before t_end
            d["t_end_deterioration"] = int(d["t_end"]) - 2
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "CaseStudyConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_results_folder(cls, folder: str, beam_type: str | None = None) -> "CaseStudyConfig":
        """Config saved by a run; falls back to defaults (with a warning) for older
        result folders that predate config.json."""
        path = os.path.join(folder, CONFIG_FILE_NAME)
        if os.path.exists(path):
            return cls.from_json(path)
        print(f"WARNING: no {CONFIG_FILE_NAME} in {folder}; assuming default case-study config.")
        cfg = cls()
        if beam_type is not None:
            cfg.beam_type = beam_type
        return cfg

    def updated(self, **overrides) -> "CaseStudyConfig":
        """Copy with the non-None overrides applied."""
        d = self.to_dict()
        d.update({k: v for k, v in overrides.items() if v is not None})
        return self.from_dict(d)


def add_config_cli_args(parser) -> None:
    """CLI flags: --config JSON plus a flag for every config field (--true-E, --p0, ...)."""
    parser.add_argument("--config", default=None,
                        help="JSON file with case-study settings (partial files are fine)")
    for f in fields(CaseStudyConfig):
        typ = f.type if isinstance(f.type, type) else {"str": str, "int": int, "float": float}[f.type]
        parser.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=typ, default=None,
                            help=f"(default: {f.default})")


def config_from_cli(args) -> CaseStudyConfig:
    cfg = CaseStudyConfig.from_json(args.config) if args.config else CaseStudyConfig()
    return cfg.updated(**{f.name: getattr(args, f.name) for f in fields(CaseStudyConfig)})
