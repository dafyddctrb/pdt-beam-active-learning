"""The physical twin: the 'real' beam with the true parameters, which produces the synthetic
measurements."""
import numpy as np

from behavior_models.beams import CantileverBeam, TwoSidedBeam


class PhysicalTwinBeam:

    def __init__(self, true_l: float, true_V: float, true_E: float, true_lambda: float, timeframe: np.ndarray,
                 beam_type: str = "cantilever", b: float = 1.0, h: float = 0.13):
        self.true_l = true_l
        self.true_V = true_V
        if beam_type == "two_sided":
            self.displacement_pmodel = TwoSidedBeam(l=true_l, V=true_V, b=b, h=h)
        elif beam_type == "cantilever":
            self.displacement_pmodel = CantileverBeam(l=true_l, V=true_V, b=b, h=h)
        else:
            raise ValueError(f"Invalid beam type: {beam_type}. Must be 'two_sided' or 'cantilever'.")
        self.true_E = true_E
        self.true_lambda = true_lambda
        self.timeframe = timeframe
        self.true_loss = self.displacement_pmodel.get_loss_at_time(self.timeframe, true_lambda)   # D(t) [%]
        self.true_I = self.displacement_pmodel.get_I_from_loss(loss=self.true_loss)
        self.true_displacement = []      # noise-free displacement of every measure_displacement call

    def measure_displacement(self, t_cur, noise_rel=0.01):
        """Displacement at time t_cur with Gaussian noise of std noise_rel x the true value."""
        loss = self.displacement_pmodel.get_loss_at_time(t=t_cur, lambda_dr=np.array([self.true_lambda]))
        I = self.displacement_pmodel.get_I_from_loss(loss=np.array([loss]))
        cur_displacement = self.displacement_pmodel.get_beam_max_deflection(I=I, E=self.true_E)
        self.true_displacement.append(cur_displacement[0][0])
        return cur_displacement + np.random.normal(loc=0, scale=noise_rel * cur_displacement)

    def measure_lambda(self, t_cur, noise_rel: float, sigma_lambda: float):
        """Measurement of lambda at year t_cur: a noisy reading of the cumulative loss D(t_cur),
        transformed back to lambda with the inverse deterioration law at the same t_cur
        (paper, Section 5.1).

        noise_rel:    std of the noise added to D, relative to the true D(t_cur).
        sigma_lambda: returned as the likelihood std of the lambda measurement.
        """
        true_loss = self.displacement_pmodel.get_loss_at_time(t=t_cur, lambda_dr=self.true_lambda)
        loss = true_loss + np.random.normal(loc=0, scale=noise_rel * true_loss)
        lambda_mean = self.displacement_pmodel.get_lambda_from_loss(t=t_cur, loss=np.array([loss]))
        return lambda_mean[0], float(sigma_lambda)
