"""Beam models: the maximum deflection of a deteriorating beam (linear elastic, Euler-Bernoulli).

Deterioration: every year the second moment of area loses lambda % of what is left,
    I(t) = I0 (1 - lambda/100)^t,  i.e. a cumulative loss D(t) = 100 (1 - (1 - lambda/100)^t) [%].
"""
import numpy as np


class BeamProperties:
    """Geometry and load of a beam with a rectangular b x h section, and the deterioration law."""

    def __init__(self, l: float, V: float, b: float = 100, h: float = 100):
        self.l = l
        self.V = V
        self.b = b
        self.h = h

    def get_I_at_time(self, lambda_dr, t_cur):
        """Second moment of area after t_cur years at the annual rate lambda_dr [%/year]."""
        return self.get_I_from_loss(loss=self.get_loss_at_time(t=t_cur, lambda_dr=lambda_dr))

    def get_I_from_loss(self, loss=0):
        """Second moment of area after a cumulative loss `loss` [%] of the intact b h^3 / 12."""
        I = (self.b * self.h ** 3) / 12
        return I * ((100 - loss) / 100)

    @staticmethod
    def get_loss_at_time(t, lambda_dr):
        """Cumulative loss D(t) [%] of the second moment of area after t years."""
        return (1 - (1 - np.asarray(lambda_dr, dtype=float) / 100) ** t) * 100

    @staticmethod
    def get_lambda_from_loss(t, loss):
        """Inverse of get_loss_at_time: the annual rate lambda [%/year] that gives the
        cumulative loss `loss` [%] after t years."""
        return (1 - (1 - np.asarray(loss, dtype=float) / 100) ** (1 / t)) * 100


class CantileverBeam(BeamProperties):
    """Cantilever under a uniformly distributed load V [kN/m]."""

    def predict(self, X):
        """Deflection for the columns (E, I) of X; the second value mirrors a surrogate's (mean, std)."""
        return self.get_beam_max_deflection(E=X[:, 0], I=X[:, 1]), 0

    def get_beam_max_deflection(self, E, I):
        return float(self.V * self.l**4) / (8 * E * I)


class TwoSidedBeam(BeamProperties):
    """Simply supported beam with a point load V [kN] at midspan."""

    def predict(self, X):
        """Deflection for the columns (E, I) of X; the second value mirrors a surrogate's (mean, std)."""
        return self.get_beam_max_deflection(E=X[:, 0], I=X[:, 1]), 0

    def get_beam_max_deflection(self, E, I):
        return (self.V * self.l**3) / (48 * E * I)
