"""Reference (exact) posterior for the cantilever-beam updating scenario.

The four updating methods (Seq/Batch x PM/SRG) all target the same posterior
given a run's full data set:

    p(E, lam | D)  ∝  p(E) p(lam)
                      * prod_k N(lam_obs_k ; lam, s_k)                 (deterioration-rate data)
                      * prod_t N(d_t ; delta(E, I(lam, t)), sigma_t)   (displacement data)

with independent Normal priors (R = I in PDTBeam). Batch updating targets it
directly; sequential updating approximates it by refitting Normal marginals after
every step. Two independent estimates of it are provided here:

  * ``reference_mcs``  — prior Monte Carlo + likelihood weighting (importance
    sampling with the prior as proposal; mean weight = evidence). Reports the
    effective sample size so a degenerate estimate is visible.
  * ``reference_grid`` — deterministic 2D quadrature on a tensor grid that is
    refined to the posterior's support (no sampling noise).

Measurements and priors are read back from a run's state-dict JSON, so the
reference uses exactly the data the methods saw.
"""
import hashlib

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from behavior_models.beams import CantileverBeam, TwoSidedBeam

from case_study_config import CaseStudyConfig

# Default scenario geometry/load (see case_study_config.py). Callers analysing a
# run pass that run's own `beam_kwargs` / `t_pred` from its saved config.
_DEFAULT_CFG = CaseStudyConfig()
SCENARIO = dict(**_DEFAULT_CFG.beam_kwargs, t_end=_DEFAULT_CFG.t_end)


def make_beam(beam_type: str = "cantilever", l=SCENARIO["l"], V=SCENARIO["V"],
              b=SCENARIO["b"], h=SCENARIO["h"]):
    if beam_type == "cantilever":
        return CantileverBeam(l=l, V=V, b=b, h=h)
    if beam_type == "two_sided":
        return TwoSidedBeam(l=l, V=V, b=b, h=h)
    raise ValueError(f"Invalid beam type: {beam_type}")


def displacement(beam, E, lam, t):
    """Physics-model deflection at time t (same mapping as PDTBeam.pmodel_predict_displacement)."""
    I = beam.get_I_at_time(lambda_dr=lam, t_cur=t)
    return np.asarray(beam.get_beam_max_deflection(E=E, I=I), dtype=float)


# --------------------------------------------------------------------------- data

def _as_list(x):
    return list(np.atleast_1d(np.asarray(x, dtype=float)))


def extract_measurements(data: dict) -> dict:
    """Priors + the full (deduplicated) measurement record of one run's state dict.

    Displacement records appear once per update (single values for sequential,
    cumulative arrays for some batch steps, and repeated on sequential replays);
    they are merged by measurement time. Deterioration-rate records are cumulative,
    so the longest one is the full set.
    """
    prior = data["0"]["0"]
    disp, prop = {}, {"mean": [], "sigma": [], "time": []}
    for t in data:
        for s in data[t]:
            st = data[t][s]
            bm = st.get("behavior_measurement") or {}
            if bm.get("mean") is not None:
                for m, sg, tt in zip(_as_list(bm["mean"]), _as_list(bm["sigma"]), _as_list(bm["time"])):
                    disp[int(tt)] = (m, sg)
            pm = st.get("property_measurement") or {}
            if pm.get("mean") is not None and len(_as_list(pm["mean"])) > len(prop["mean"]):
                prop = {k: _as_list(pm[k]) for k in ("mean", "sigma", "time")}
    times = sorted(disp)
    return {
        "E_prior": (float(prior["E"]["mu"]), float(prior["E"]["sigma"])),
        "lambda_prior": (float(prior["lambda"]["mu"]), float(prior["lambda"]["sigma"])),
        "disp_time": np.array(times, dtype=float),
        "disp_mean": np.array([disp[t][0] for t in times]),
        "disp_sigma": np.array([disp[t][1] for t in times]),
        "prop_mean": np.array(prop["mean"]),
        "prop_sigma": np.array(prop["sigma"]),
    }


def measurement_key(meas: dict) -> str:
    """Fingerprint of a data set; runs sharing it (one repetition) share one reference."""
    vals = np.concatenate([meas["disp_time"], meas["disp_mean"], meas["disp_sigma"],
                           meas["prop_mean"], meas["prop_sigma"],
                           meas["E_prior"], meas["lambda_prior"]])
    return hashlib.sha1(np.round(vals, 12).tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------- likelihood

def log_likelihood(beam, E, lam, meas):
    """Log-likelihood of all displacement + deterioration-rate data (vectorised)."""
    E = np.asarray(E, dtype=float)
    lam = np.asarray(lam, dtype=float)
    ll = np.zeros(np.broadcast(E, lam).shape)
    for t, m, s in zip(meas["disp_time"], meas["disp_mean"], meas["disp_sigma"]):
        ll += norm.logpdf(m, loc=displacement(beam, E, lam, t), scale=s)
    for m, s in zip(meas["prop_mean"], meas["prop_sigma"]):
        ll += norm.logpdf(m, loc=lam, scale=s)
    return ll


def log_prior(E, lam, meas):
    return (norm.logpdf(E, *meas["E_prior"]) + norm.logpdf(lam, *meas["lambda_prior"]))


# --------------------------------------------------------------- weighted summary

def _weighted_summary(x, w):
    mu = float(np.sum(w * x))
    return mu, float(np.sqrt(np.sum(w * (x - mu) ** 2)))


def _summarise(E, lam, d, w):
    out = {}
    for name, x in (("E", E), ("lambda", lam), ("displacement", d)):
        out[f"{name}_mean"], out[f"{name}_std"] = _weighted_summary(x, w)
    return out


# ---------------------------------------------------------------------------- MCS

def reference_mcs(meas, beam_type="cantilever", n=2_000_000, chunk=500_000,
                  n_resample=1000, t_pred=SCENARIO["t_end"], seed=0, beam_kwargs=None):
    """Prior-sampling Monte Carlo reference (self-normalised importance sampling).

    Returns posterior moments, log-evidence, ESS, and ``n_resample`` posterior
    draws (multinomial resampling of the weighted prior sample) for sample-based
    comparison metrics.
    """
    beam = make_beam(beam_type, **(beam_kwargs or {}))
    rng = np.random.default_rng(seed)
    E_all, lam_all, logw_all = [], [], []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        E = rng.normal(*meas["E_prior"], size=m)
        lam = rng.normal(*meas["lambda_prior"], size=m)
        E_all.append(E)
        lam_all.append(lam)
        logw_all.append(log_likelihood(beam, E, lam, meas))
    E, lam, logw = (np.concatenate(a) for a in (E_all, lam_all, logw_all))

    log_evidence = float(logsumexp(logw) - np.log(n))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))

    d = displacement(beam, E, lam, t_pred)
    out = _summarise(E, lam, d, w)
    idx = rng.choice(n, size=n_resample, replace=True, p=w)
    out.update(method="mcs", n=n, ess=ess, log_evidence=log_evidence,
               E_samples=E[idx], lambda_samples=lam[idx], displacement_samples=d[idx])
    return out


# --------------------------------------------------------------------------- grid

def _grid_bounds_from_mass(x, mass, tol):
    keep = np.nonzero(mass > tol * mass.max())[0]
    lo, hi = x[keep[0]], x[keep[-1]]
    pad = 0.15 * (hi - lo) + (x[1] - x[0])
    return lo - pad, hi + pad


def _grid_posterior(beam, meas, E_lim, lam_lim, n):
    E = np.linspace(*E_lim, n)
    lam = np.linspace(*lam_lim, n)
    EE, LL = np.meshgrid(E, lam, indexing="ij")
    logp = log_prior(EE, LL, meas) + log_likelihood(beam, EE, LL, meas)
    return E, lam, EE, LL, logp


def reference_grid(meas, beam_type="cantilever", n=800, n_std=8.0, tol=1e-12,
                   n_resample=1000, t_pred=SCENARIO["t_end"], seed=0, beam_kwargs=None):
    """Tensor-grid quadrature reference, refined to the posterior support.

    Pass 1 covers prior mean ± ``n_std`` prior sd; pass 2 re-grids the box where
    the pass-1 marginals exceed ``tol`` x their peak, so the n x n nodes resolve
    the (much narrower) posterior. Posterior draws are taken by sampling grid
    cells by weight and jittering uniformly within the cell.
    """
    beam = make_beam(beam_type, **(beam_kwargs or {}))
    (Em, Es), (lm, ls) = meas["E_prior"], meas["lambda_prior"]
    E_lim, lam_lim = (Em - n_std * Es, Em + n_std * Es), (lm - n_std * ls, lm + n_std * ls)

    E, lam, _, _, logp = _grid_posterior(beam, meas, E_lim, lam_lim, n)
    p = np.exp(logp - logp.max())
    E_lim = _grid_bounds_from_mass(E, p.sum(axis=1), tol)
    lam_lim = _grid_bounds_from_mass(lam, p.sum(axis=0), tol)

    E, lam, EE, LL, logp = _grid_posterior(beam, meas, E_lim, lam_lim, n)
    dE, dlam = E[1] - E[0], lam[1] - lam[0]
    log_evidence = float(logsumexp(logp) + np.log(dE * dlam))
    w = np.exp(logp - logp.max())
    w /= w.sum()

    d = displacement(beam, EE, LL, t_pred)
    out = _summarise(EE, LL, d, w)

    rng = np.random.default_rng(seed)
    idx = rng.choice(w.size, size=n_resample, replace=True, p=w.ravel())
    Es_ = EE.ravel()[idx] + rng.uniform(-0.5, 0.5, n_resample) * dE
    ls_ = LL.ravel()[idx] + rng.uniform(-0.5, 0.5, n_resample) * dlam
    out.update(method="grid", n=n * n, log_evidence=log_evidence,
               E_grid=E, lambda_grid=lam,
               E_marginal=w.sum(axis=1), lambda_marginal=w.sum(axis=0),
               node_E=EE.ravel(), node_lambda=LL.ravel(), node_displacement=d.ravel(),
               node_weight=w.ravel(),
               E_samples=Es_, lambda_samples=ls_,
               displacement_samples=displacement(beam, Es_, ls_, t_pred))
    return out


# ------------------------------------------------------------------------ metrics

def _energy_distance_2d(X, Y):
    """Energy distance between two 2D samples: 2E|X-Y| - E|X-X'| - E|Y-Y'| (>= 0)."""
    from scipy.spatial.distance import cdist
    return float(2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())


def compare_to_reference(E, lam, d, ref, grid_ref=None, n_joint=1000, seed=0):
    """Accuracy of one posterior sample set against a reference posterior.

    Per quantity (E, lambda, displacement): standardised mean error
    (m - m_ref)/sd_ref, sd ratio sd/sd_ref, and 1-Wasserstein distance divided by
    sd_ref. Joint: energy distance between (E, lambda) samples standardised by the
    reference sd. Wasserstein uses the exact grid nodes/weights when ``grid_ref``
    is given, otherwise the reference's posterior draws.
    """
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    E, lam, d = (np.asarray(x, dtype=float).ravel() for x in (E, lam, d))
    out = {}
    for name, x in (("E", E), ("lambda", lam), ("displacement", d)):
        mu_r, sd_r = ref[f"{name}_mean"], ref[f"{name}_std"]
        out[f"{name}_mean_err"] = (np.mean(x) - mu_r) / sd_r
        out[f"{name}_std_ratio"] = np.std(x) / sd_r
        if grid_ref is not None:
            w1 = wasserstein_distance(x, grid_ref[f"node_{name}"], v_weights=grid_ref["node_weight"])
        else:
            w1 = wasserstein_distance(x, ref[f"{name}_samples"])
        out[f"{name}_w1"] = w1 / sd_r

    scale = np.array([ref["E_std"], ref["lambda_std"]])
    shift = np.array([ref["E_mean"], ref["lambda_mean"]])
    X = np.column_stack((E, lam))
    Y = np.column_stack((ref["E_samples"], ref["lambda_samples"]))
    if len(X) > n_joint:
        X = X[rng.choice(len(X), n_joint, replace=False)]
    if len(Y) > n_joint:
        Y = Y[rng.choice(len(Y), n_joint, replace=False)]
    out["joint_energy"] = _energy_distance_2d((X - shift) / scale, (Y - shift) / scale)
    return out


def pooled_marginal_density(grids, quantity, n=600):
    """Exact reference marginal density of E, lambda or displacement, averaged over
    several data sets (each grid reference contributes equally) on one common axis.

    E and lambda use the grid's quadrature marginals directly; the displacement
    density is the weighted histogram of the deflection at every grid node (a
    deterministic push-forward of the grid weights, no sampling). Returns
    ``(y, density, mean)``.
    """
    if quantity == "displacement":
        lo = min(g["node_displacement"][g["node_weight"] > 1e-12].min() for g in grids)
        hi = max(g["node_displacement"][g["node_weight"] > 1e-12].max() for g in grids)
        edges = np.linspace(lo, hi, n + 1)
        y = 0.5 * (edges[:-1] + edges[1:])
        dens = np.mean([np.histogram(g["node_displacement"], bins=edges,
                                     weights=g["node_weight"], density=True)[0] for g in grids], axis=0)
    else:
        axis, marg = f"{quantity}_grid", f"{quantity}_marginal"
        y = np.linspace(min(g[axis][0] for g in grids), max(g[axis][-1] for g in grids), n)
        dens = np.mean([np.interp(y, g[axis], g[marg] / (g[axis][1] - g[axis][0]), left=0, right=0)
                        for g in grids], axis=0)
    mean = float(np.mean([g[f"{quantity}_mean"] for g in grids]))
    return y, dens, mean
