import numpy as np
from typing import List, Union
from scipy.stats import norm


class MeasurementLikelihood:
    """Gaussian likelihood of measured responses (e.g. displacements) at several time steps."""

    def __init__(self):
        self.measured_mean = None
        self.measured_sigma = None
        self.measured_time = None
        self.last_prediction = None     # responses predicted in the last log_likelihood call

    def set_measured_mean_and_sigma(self, measured_mean: np.ndarray, 
                                    measured_sigma: Union[float, List[float]],
                                    measured_time: np.ndarray):
        self.measured_mean = measured_mean
        if self.measured_mean.ndim == 1:
            self.measured_mean = self.measured_mean.reshape(-1, 1)
        # One sigma row per measurement: a 1-D vector [s1..sK] is a column (one
        # per measurement), NOT a row to be tiled — atleast_2d would make it
        # (1, K) and the repeat below would silently tile it to (K, K).
        self.measured_sigma = np.asarray(measured_sigma, dtype=float)
        if self.measured_sigma.ndim <= 1:
            self.measured_sigma = self.measured_sigma.reshape(-1, 1)
        if self.measured_sigma.shape[0] == 1 and self.measured_mean.shape[0] > 1:
            self.measured_sigma = np.repeat(self.measured_sigma, self.measured_mean.shape[0], axis=0)
        self.measured_sigma = np.abs(self.measured_sigma)
        self.measured_time = measured_time

    @staticmethod
    def likelihood_of_prediction(prediction: np.ndarray, measured_mean: np.ndarray, measured_sigma: np.ndarray) -> np.ndarray:
        """Likelihood of predicted responses given the measurements of one time step: an equally
        weighted mixture of Gaussians, one per measurement (a single Gaussian for one reading)."""
        n = measured_mean.shape[0]
        w = 1 / n
        likelihood = np.zeros(prediction.shape)
        for i in range(n):
            likelihood += w * norm.pdf(prediction, measured_mean[i], measured_sigma[i])
        return likelihood

    def log_likelihood(self, parameters: np.ndarray, prediction_model,
                       return_std: bool = False, second_order: bool = False):
        """
        Compute the log likelihood of `parameters`, mapping parameters -> response at each measured time via
        `prediction_model`.

        With `return_std`, also propagate the surrogate predictive std to the std
        of the log-likelihood and return `(log_likelihood, sigma_logL)`. The
        surrogate posterior `(mean, std)` enters via a first- (default) or
        second-order delta method; the per-time sensitivity d logL/d delta is a
        central finite difference of the SAME per-time likelihood used for the
        mean, so it never drifts from the likelihood and follows any change to it:

            var_logL = sum_t (d logL_t/d delta)^2 * std_t^2   [+ 0.5 (d2 logL_t)^2 std_t^4]

        The default path (`return_std=False`) is unchanged: it does not request the
        std from `prediction_model` and returns only `log_likelihood`.
        """
        if parameters.ndim == 1:
            parameters = parameters.reshape(-1, 1).T
        likelihood = np.ones((parameters.shape[0], 1))
        var = np.zeros((parameters.shape[0], 1))
        mean_corr = np.zeros((parameters.shape[0], 1))   # second-order log-mean correction

        for i_t_cur, t_cur in enumerate(self.measured_time):
            mm = self.measured_mean[i_t_cur, :]
            ms = self.measured_sigma[i_t_cur, :]
            if return_std:
                self.last_prediction, pred_std = prediction_model(parameters, t_cur=t_cur, return_std=True)
            else:
                self.last_prediction, _ = prediction_model(parameters, t_cur=t_cur)
            cur_likelihood = self.likelihood_of_prediction(self.last_prediction, mm, ms)
            likelihood *= cur_likelihood

            if return_std:
                e = max(1e-3 * float(ms[0]), 1e-6)
                logL0 = np.log(np.clip(cur_likelihood, 1e-300, None))
                logLp = np.log(np.clip(self.likelihood_of_prediction(self.last_prediction + e, mm, ms), 1e-300, None))
                logLm = np.log(np.clip(self.likelihood_of_prediction(self.last_prediction - e, mm, ms), 1e-300, None))
                s = np.asarray(pred_std).reshape(logL0.shape)
                g1 = (logLp - logLm) / (2.0 * e)               # d logL / d delta
                g2 = (logLp - 2.0 * logL0 + logLm) / (e ** 2)  # d^2 logL / d delta^2
                var += g1 ** 2 * s ** 2
                if second_order:
                    var += 0.5 * g2 ** 2 * s ** 4
                    mean_corr += 0.5 * g2 * s ** 2

        log_likelihood = np.log(likelihood)
        if return_std:
            return log_likelihood + mean_corr, np.sqrt(var)
        return log_likelihood
