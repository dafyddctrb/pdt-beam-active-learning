import numpy as np
from typing import List, Union
from behavior_models.beams import TwoSidedBeam, CantileverBeam
from behavior_models.gp_surrogate import SRGModel
from updating.ERADist import ERADist
from updating.ERANataf import ERANataf
from updating.likelihood_functions import MeasurementLikelihood
from pathlib import Path

class PDTBeam:

    def __init__(self, l: float, V: float, timeframe: np.ndarray,
                 E: float, E_cov: float, lambda_mean: float, lambda_cov: float,
                 b: float = 1.0, h: float = 0.13, beam_type: str = "cantilever",
                 N_samples: int = 1000):
        """Digital twin with normal priors E ~ N(E, E_cov E) and lambda ~ N(lambda_mean,
        lambda_cov lambda_mean); N_samples prior samples are stored at t = 0."""

        self.N_samples = N_samples
        self.l = l
        self.V = V
        self.b = b
        self.h = h
        self.timeframe = timeframe
        self.nr_likelihood_calls = 0
        self.sublevel_timestep = 0
        self.beam_type = beam_type
        self.initialize_displacement()

        self.cur_timestep = 5
        self.E_distribution = ERADist("normal", "PAR", [E, E_cov*E])
        self.lambda_distribution = ERADist("normal", "PAR", [lambda_mean, lambda_cov*lambda_mean])
        self.R = np.eye(2)
        self.joint_distribution = ERANataf([self.E_distribution, self.lambda_distribution], self.R)
        self.state_dict = {}
        # Per-timestep held-out surrogate-accuracy trajectory. `_holdout` is an
        # optional (X_test, y_test) injected by an evaluation harness; when present,
        # record_accuracy_snapshot() appends one entry per timestep so accuracy gain
        # per training point can be traced over the campaign (first entry = pre-AL).
        self.accuracy_history = []
        self._holdout = None

        # Parameter registry: centralised metadata for each random variable
        self.param_registry = {
            'E':         {'distribution': self.E_distribution,      'learnable': True, 'surrogate_feature': True},
            'lambda_dr': {'distribution': self.lambda_distribution, 'learnable': True, 'surrogate_feature': True},
        }
        self._surrogate_cols = [i for i, k in enumerate(self.param_registry) if self.param_registry[k]['surrogate_feature']]


    def set_parameters(self, plot_dir: Path = None, model_type: str = "pmodel"):
        self.model_type = model_type
        self.plot_dir = plot_dir

        self.displacement_srg = SRGModel(self.displacement_pmodel.get_beam_max_deflection, plot_dir=plot_dir)
        self.displacement_l = MeasurementLikelihood()
        # only the surrogate methods need (and pay for) the initial training design
        if model_type == "srg":
            self.initial_displacement_srg_model()
        self.save_state_dict(t_cur=0)

    def _score_surrogate_on(self, X_test, y_test):
        """RMSE + NRMSE (RMSE/mean|true|) of the current surrogate on (X_test, y_test)."""
        try:
            y_pred, _ = self.displacement_srg.predict(np.asarray(X_test, dtype=float), return_std=False)
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
            y_true = np.asarray(y_test, dtype=float).reshape(-1)
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            denom = float(np.mean(np.abs(y_true))) or 1.0
            return rmse, rmse / denom
        except Exception as e:
            print(f"  [accuracy score] skipped: {e}")
            return float('nan'), float('nan')

    def _local_test_set(self, t_cur, n_local=300, seed=777):
        """Decision-relevant test set: the CURRENT posterior region at this timestep's
        deterioration state. Samples [E, lambda] from the (updated) marginals, maps
        lambda -> I at t_cur, and evaluates the true deflection. A fixed seed (with
        global RNG state saved/restored so the campaign's stream is untouched) keeps
        the quantile positions stable across timesteps/methods, so only the posterior
        itself moves."""
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            E = np.asarray(self.E_distribution.random(n_local)).reshape(-1)
            lam = np.asarray(self.lambda_distribution.random(n_local)).reshape(-1)
        finally:
            np.random.set_state(rng_state)
        t = int(t_cur) if int(t_cur) >= int(self.timeframe[0]) else int(self.timeframe[0])
        I = np.asarray(self.displacement_pmodel.get_I_at_time(
            lambda_dr=lam, t_cur=t)).reshape(-1)
        X = np.column_stack((E, I))
        y_true = np.asarray(self.displacement_pmodel.get_beam_max_deflection(E=E, I=I)).reshape(-1)
        return X, y_true

    def record_accuracy_snapshot(self, t_cur):
        """Score the CURRENT displacement surrogate and append a snapshot. No-op unless
        `_holdout` is set, so production runs are unaffected. Captures TWO accuracies:
          - global  (rmse/nrmse): the fixed injected held-out set (prior x deterioration);
          - local   (rmse_local/nrmse_local): the current posterior region at this
            timestep (the decision-relevant zone the surrogate is enriched on).
        Used to trace accuracy gain vs training points added across the campaign."""
        holdout = getattr(self, "_holdout", None)
        if holdout is None:
            return
        try:
            n_train = int(self.displacement_srg.X_train_current.shape[0])
        except Exception:
            n_train = None
        rmse_g, nrmse_g = self._score_surrogate_on(*holdout)
        try:
            X_loc, y_loc = self._local_test_set(t_cur)
            rmse_l, nrmse_l = self._score_surrogate_on(X_loc, y_loc)
        except Exception as e:
            print(f"  [local accuracy t={t_cur}] skipped: {e}")
            rmse_l, nrmse_l = float('nan'), float('nan')
        self.accuracy_history.append({
            "t": int(t_cur), "n_train": n_train, "calls": int(self.nr_likelihood_calls),
            "rmse": rmse_g, "nrmse": nrmse_g, "rmse_local": rmse_l, "nrmse_local": nrmse_l,
        })

    def save_state_dict(self, t_cur: int = 0, E_samples: np.ndarray = None, lambda_samples: np.ndarray = None, 
                        I_samples: np.ndarray = None, displacement_samples:np.ndarray = None, 
                        behavior_measurement_dict: dict = None, property_measurement_dict: dict = None, state_type: str = "initial"):
        if E_samples is None:
            E_samples = self.E_distribution.random(self.N_samples)
        if lambda_samples is None:
            lambda_samples = self.lambda_distribution.random(self.N_samples)
        if I_samples is None:
            I_samples = self.displacement_pmodel.get_I_at_time(lambda_dr=lambda_samples, t_cur=t_cur)
        if displacement_samples is None:
            displacement_samples = self.predict_displacement(E_samples, I_samples)[0]
            # check if displacement_sampels is a list
            if not isinstance(displacement_samples, list):
                displacement_samples = [displacement_samples]

        if behavior_measurement_dict is None:
            behavior_measurement_dict = {"measurement": {"mean": None, "sigma": None}}
        if property_measurement_dict is None:
            property_measurement_dict = {"measurement": {"mean": None, "sigma": None}}


        if t_cur not in self.state_dict:
            self.state_dict[t_cur] = {}

        self.state_dict[t_cur][self.sublevel_timestep] = {
            "state_type": state_type,
            "E": {"type": "normal", "mu": self.E_distribution.Par['mu'], "sigma": self.E_distribution.Par['sigma'], "samples": E_samples},
            "lambda": {"type": "normal", "mu": self.lambda_distribution.Par['mu'], "sigma": self.lambda_distribution.Par['sigma'], "samples": lambda_samples},
            "I": {"type": "normal", "mu": np.mean(I_samples), "sigma": np.std(I_samples), "samples": I_samples},
            "displacement": {"type": "normal", "mu": np.mean(displacement_samples), "sigma": np.std(displacement_samples), "samples": displacement_samples},
            "behavior_measurement": behavior_measurement_dict["measurement"],
            "property_measurement": property_measurement_dict["measurement"],
            "nr_likelihood_calls": self.nr_likelihood_calls,
            # Surrogate AL training set (E, I), seed-first then AL-appended, so the
            # acquisition storyboard can show which points each method selected.
            # None for physics-based methods (no surrogate).
            "surrogate_train_X": (self.displacement_srg.X_train_current.tolist()
                                  if self.model_type == "srg"
                                  and self.displacement_srg.X_train_current is not None
                                  else None),
            "surrogate_n_init_train": int(getattr(self, "n_init_train", 0)),
        }
        
        self.sublevel_timestep += 1

    def predict_displacement(self, E_samples: np.ndarray, lambda_samples: np.ndarray):
        if self.model_type == "pmodel":
            return self.displacement_pmodel.predict(np.column_stack((np.array(E_samples), np.array(lambda_samples))))
        elif self.model_type == "srg":
            return self.displacement_srg.predict(np.column_stack((np.array(E_samples), np.array(lambda_samples))))
        else:
            raise ValueError(f"Invalid model type: {self.model_type}. Must be 'pmodel' or 'srg'.")

    def initialize_displacement(self):
        if self.beam_type == "two_sided":
            self.displacement_pmodel = TwoSidedBeam(l=self.l, V=self.V, b=self.b, h=self.h)
        elif self.beam_type == "cantilever":
            self.displacement_pmodel = CantileverBeam(l=self.l, V=self.V, b=self.b, h=self.h)
        else:
            raise ValueError(f"Invalid beam type: {self.beam_type}. Must be 'two_sided' or 'cantilever'.")


    def initial_displacement_srg_model(self):
        nr_samples = max(len(self.joint_distribution.Marginals) * 2, 5)
        self.nr_likelihood_calls += nr_samples
        scale_lower = 0.7
        scale_upper = 1.3
        E_uniform_distribution = ERADist("uniform", "PAR", [self.E_distribution.Par['mu']*scale_lower, self.E_distribution.Par['mu'] * scale_upper])
        lambda_uniform_distribution = ERADist("uniform", "PAR", [0.06, 0.16])
        
        E_samples = E_uniform_distribution.random(nr_samples)
        lambda_samples = lambda_uniform_distribution.random(nr_samples)
        I_samples = self.displacement_pmodel.get_I_at_time(lambda_dr=lambda_samples, t_cur=self.timeframe[0])
        X_train_current = np.column_stack((E_samples, I_samples))
        # Seed-design size: the first n_init_train rows of X_train_current are this
        # initial design; everything appended later is AL-selected (used by the
        # acquisition storyboard to split seed vs AL points).
        self.n_init_train = int(X_train_current.shape[0])
        self.displacement_srg.train_model(X_train_current)
        # The initial evaluations are already counted on this ledger (nr_samples above);
        # zero the SRG ledger so the first retrain sweep doesn't count them again.
        self.displacement_srg.likelihood_calls = 0

    def pmodel_predict_displacement(self, X_candidate_points: np.ndarray, t_cur: int = None):
        # if X_candidate_points is a 1D array, reshape it to a 2D array
        if X_candidate_points.ndim == 1:
            X_candidate_points = np.array([X_candidate_points])
        
        if t_cur is None:
            t_cur = self.cur_timestep
        E_samples = X_candidate_points[:, 0]
        I_samples = self.displacement_pmodel.get_I_at_time(lambda_dr=X_candidate_points[:, 1], t_cur=t_cur)
        X_candidate_points = np.column_stack((E_samples, I_samples))
        self.nr_likelihood_calls += X_candidate_points.shape[0]
        return self.displacement_pmodel.predict(X_candidate_points)[0].reshape(-1,1), np.ones(X_candidate_points.shape[0])


    def srg_predict_displacement(self, X_candidate_points: np.ndarray, t_cur: int = None, return_std: bool = False):
        # if X_candidate_points is a 1D array, reshape it to a 2D array
        if X_candidate_points.ndim == 1:
            X_candidate_points = np.array([X_candidate_points])

        if t_cur is None:
            t_cur = self.cur_timestep
        E_samples = X_candidate_points[:, 0]
        I_samples = self.displacement_pmodel.get_I_at_time(lambda_dr=X_candidate_points[:, 1], t_cur=t_cur)
        X_candidate_points = np.column_stack((E_samples, I_samples))
        # Select only surrogate feature columns
        X_surrogate = X_candidate_points[:, self._surrogate_cols]
        # The Gaussian likelihood discards the std, so default to the mean-only
        # (variance-skipping) path — this is the dominant cost in the SuS/aCS loops.
        return self.displacement_srg.predict(X_surrogate, return_std=return_std)


    def srg_retrain_model(self, l_class, X_candidate_points: np.ndarray, p_samples: np.ndarray, c: float, p0: float = None, **al_kwargs):
        if X_candidate_points.ndim == 1:
            X_candidate_points = X_candidate_points.reshape(-1, 1)
        E_samples = X_candidate_points[:, 0]
        lambda_samples = X_candidate_points[:, 1]
        # One candidate slice per measurement time: slice k holds (E, I(lambda, t_k)),
        # so the AL scores the SAME product log-likelihood the SuS loop evaluates
        # (sequential updating sets a single measurement -> K=1).
        measured_times = np.atleast_1d(self.displacement_l.measured_time)
        X_slices = np.stack([
            np.column_stack((E_samples,
                             self.displacement_pmodel.get_I_at_time(
                                 lambda_dr=lambda_samples, t_cur=t_k)))[:, self._surrogate_cols]
            for t_k in measured_times
        ])
        # The acquisition evaluates the log-likelihood through l_class itself, mapping
        # parameter candidates -> response via the surrogate predictor; the slices only
        # supply the surrogate-feature vector for the new training point. Forward the
        # BUS quantities: max log-likelihood c (=logl_hat) and the SuS conditional
        # probability p0 (the level is recomputed from it), plus the per-candidate
        # BUS auxiliary p (-> ln p in the LSF mean).
        self.displacement_srg.retrain_model(
            l_class=l_class,
            prediction_model=self.srg_predict_displacement,
            X_candidate_params=X_candidate_points,
            X_candidate_slices=X_slices,
            bus_logl_hat=c,
            bus_p_samples=p_samples,
            bus_p0=p0,
            **al_kwargs)
        # instrumentation: record this AL invocation (points added = likelihood_calls
        # accrued in retrain_model, before reset; total = current train-set size).
        if not hasattr(self, 'al_trace_records'):
            self.al_trace_records = []
        self.al_trace_records.append({
            'timestep': int(self.cur_timestep),
            'p0': float(p0) if p0 is not None else None,
            'n_added': int(self.displacement_srg.likelihood_calls),
            'n_train_total': int(len(self.displacement_srg.X_train_current)),
            'min_u_trace': [round(x, 3) for x in getattr(self.displacement_srg, 'last_min_u_trace', [])],
        })
        self.nr_likelihood_calls += self.displacement_srg.likelihood_calls
        self.displacement_srg.likelihood_calls = 0


    def set_measured_displacement(self, measured_displacement_mean: np.ndarray, 
                                        measured_displacement_sigma: Union[float, List[float]],
                                        measured_displacement_time: np.ndarray):
        self.displacement_l.set_measured_mean_and_sigma(measured_displacement_mean, measured_displacement_sigma, measured_displacement_time)

    def update_distributions(self, samplesX: np.ndarray, displacement_samples: np.ndarray, measurement_dict: dict, t_cur: int = 0):
        self.E_distribution = ERADist("normal", "PAR", [np.mean(samplesX[:, 0]), np.std(samplesX[:, 0])])
        self.lambda_distribution = ERADist("normal", "PAR", [np.mean(samplesX[:, 1]), np.std(samplesX[:, 1])])
        # Update registry references
        self.param_registry['E']['distribution'] = self.E_distribution
        self.param_registry['lambda_dr']['distribution'] = self.lambda_distribution
        # Reconstruct joint distribution from registry
        self.joint_distribution = ERANataf(
            [self.param_registry[k]['distribution'] for k in self.param_registry],
            self.R
        )
        I_samples = self.displacement_pmodel.get_I_at_time(lambda_dr=samplesX[:, 1], t_cur=t_cur)
        print("saving state dict")
        self.save_state_dict(t_cur=t_cur, E_samples=samplesX[:, 0], 
                            lambda_samples=samplesX[:, 1], I_samples=I_samples, 
                            displacement_samples=displacement_samples, behavior_measurement_dict=measurement_dict, state_type="behavior_updating")
        

    def update_deterioration_rate_distribution(self, new_lambda_distribution: ERADist, t_cur: int = 0, measurement_dict: dict = None):
        self.lambda_distribution = new_lambda_distribution
        self.param_registry['lambda_dr']['distribution'] = self.lambda_distribution
        self.joint_distribution = ERANataf(
            [self.param_registry[k]['distribution'] for k in self.param_registry],
            self.R
        )
        self.save_state_dict(t_cur=t_cur, state_type="property_direct_update", property_measurement_dict=measurement_dict)
