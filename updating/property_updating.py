"""Property updating: the posterior of a normally distributed parameter given direct, noisy
observations of it, by numerical integration on a grid (used for lambda and the deterioration
readings)."""
import numpy as np
import scipy.stats as stats

from updating.ERADist import ERADist


def update_normal_by_integration(prior_dist: ERADist, observations: np.ndarray,
                                                    observation_stds: np.ndarray,
                                                    n_grid_points: int = 10000,
                                                    n_std_range: float = 6.0) -> ERADist:
    """
    Updates a normal ERADist distribution with an array of observations using numerical integration.

    This function computes the posterior distribution by numerically evaluating the
    unnormalized posterior on a grid and then fitting a normal distribution to the result.

    Args:
        prior_dist (ERADist): The prior normal distribution.
        observations (np.ndarray): An array of observed values.
        observation_stds (np.ndarray): An array of standard deviations for each observation error.
        n_grid_points (int): Number of grid points for numerical integration (default: 10000).
        n_std_range (float): Number of standard deviations to extend the grid (default: 6.0).

    Returns:
        ERADist: The posterior normal distribution.

    Raises:
        ValueError: If the prior distribution is not a normal distribution, or if the
                    shapes of observations and observation_stds do not match.
    """
    if prior_dist.Name not in ['normal', 'gaussian']:
        raise ValueError("Prior distribution must be normal or gaussian.")

    if observations.shape != observation_stds.shape:
        raise ValueError("Shape of observations and observation_stds must be the same.")

    prior_mean = prior_dist.Par['mu']
    prior_std = prior_dist.Par['sigma']

    # Estimate a reasonable range for the grid based on prior and observations
    obs_mean = np.mean(observations)
    grid_center = (prior_mean + obs_mean) / 2
    grid_spread = max(prior_std, np.std(observations), np.max(observation_stds)) * n_std_range

    theta_grid = np.linspace(grid_center - grid_spread, grid_center + grid_spread, n_grid_points)
    d_theta = theta_grid[1] - theta_grid[0]

    # Compute log prior on the grid
    log_prior = stats.norm.logpdf(theta_grid, loc=prior_mean, scale=prior_std)

    # Compute log likelihood for all observations
    log_likelihood = np.zeros_like(theta_grid)
    for obs, obs_std in zip(observations, observation_stds):
        log_likelihood += stats.norm.logpdf(obs, loc=theta_grid, scale=obs_std)

    # Compute log unnormalized posterior
    log_posterior_unnorm = log_prior + log_likelihood

    # Normalize using log-sum-exp trick for numerical stability
    log_posterior_max = np.max(log_posterior_unnorm)
    posterior_unnorm = np.exp(log_posterior_unnorm - log_posterior_max)
    normalizing_constant = np.sum(posterior_unnorm) * d_theta
    posterior = posterior_unnorm / normalizing_constant

    # Compute posterior mean and standard deviation from the grid
    posterior_mean = np.sum(theta_grid * posterior) * d_theta
    posterior_var = np.sum((theta_grid - posterior_mean) ** 2 * posterior) * d_theta
    posterior_std = np.sqrt(posterior_var)

    return ERADist('normal', 'PAR', [posterior_mean, posterior_std])
