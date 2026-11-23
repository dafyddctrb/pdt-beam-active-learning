import numpy as np
import scipy as sp
np.seterr(all='ignore')

# Make sure to include aCS.py in your workspace
from .AL_aCS_aBUS import AL_aCS_aBUS

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from .ERADist import ERADist
from .ERANataf import ERANataf

"""
---------------------------------------------------------------------------
adaptive BUS with subset simulation and active learning
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Matthias Willer
Felipe Uribe
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Modified by:
Dafydd Cotoarbă (TUM Georg Nemetschek Institute and Engineering Risk Analysis Group, TUM), 2025-2026
* GP surrogate prediction model in place of the forward model
* active learning (surrogate retraining) at each subset level
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
* Use of log-evidence
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* logcE    : model log-evidence
* c        : scaling constant that holds 1/c >= Lmax
* sigma    : last spread of the proposal
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""

def AL_aBUS_SuS(N, p0, l_class, srg_prediction_model, retrain_srg_model, distr, active_learning: bool = False, max_it: int = 20, al_kwargs: dict = None):
    if (N*p0 != np.fix(N*p0)) or (1/p0 != np.fix(1/p0)):
        raise RuntimeError('N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly')

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):    # use Nataf transform (dependence)
        n   = len(distr.Marginals)+1   # number of random variables + p Uniform variable of BUS
        u2x = lambda u: distr.U2X(u)   # from u to x

    elif isinstance(distr, ERADist):   # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or use an ERANataf object
        n   = 2               # number of random variables + p Uniform variable of BUS
        u2x = lambda u: distr.icdf(sp.stats.norm.cdf(u))   # from u to x
    else:
        raise RuntimeError('Incorrect distribution. Please create an ERADist/Nataf object!')

    # limit state funtion for the observation event (Ref.1 Eq.12)
    # log likelihood in standard space

    # LSF
    h_LSF = lambda pi_u, logl_hat, log_L: np.log(sp.stats.norm.cdf(pi_u)) + logl_hat - log_L
    # note that h_LSF = log(pi) + l(i) - leval
    # where pi = normcdf(u_j(end,:)) is the standard uniform variable of BUS

    # initialization of variables
    i        = 0                         # number of conditional level
    lam      = 0.6                       # initial scaling parameter \in (0,1)
    samplesU = {'seeds': list(),
                'total': list()}
    samplesX = list()
    #
    geval = np.empty(N)                # space for the LSF evaluations
    leval = np.empty(N)                # space for the LSF evaluations
    cur_predictions = np.empty(N)
    predictions = np.empty((max_it+1, N)) # space for the displacements
    h     = np.empty(max_it)           # space for the intermediate leveles
    prob  = np.empty(max_it)           # space for the failure probability at each level

    # aBUS-SuS procedure
    # initial MCS step
    print('Evaluating log-likelihood function ...\t', end='')
    u_j   = np.random.normal(size=(n,N))  # N samples from the prior distribution
    

    leval = l_class.log_likelihood(u2x(u_j[0:-1,:].T), srg_prediction_model).flatten()
    cur_predictions = l_class.last_prediction.flatten()

    print('Done!')
    logl_hat = max(leval)   # =-log(c) (Ref.1 Alg.5 Part.3)
    print('Initial maximum log-likelihood: ', logl_hat)

    # Both acquisition families refine at the top of the level loop, on the full
    # population. BUS-space acquisitions recompute their own (logl_hat, h_i) target
    # from p0 each AL iteration (self-consistent moving target), so after the retrain
    # the population leval is re-evaluated with the refined surrogate — the level
    # (geval -> h_i -> seeds) and aCS then all run against the same refined model.
    is_bus = str((al_kwargs or {}).get('acquisition_function', '')).startswith('bus_')

    i = 0
    # SuS stage
    h[i] = np.inf
    while i < max_it:
        # active learning (output-space): refine on the full population; the
        # displacement-space limit state g = meas - pred needs no SuS threshold.
        if active_learning and retrain_srg_model is not None and not is_bus: # and nF > 0:
            retrain_srg_model(l_class=l_class,
                              X_candidate_points=u2x(u_j[0:-1,:].T),
                              p_samples=sp.stats.norm.cdf(u_j[-1,:]),
                              c=logl_hat,
                              **(al_kwargs or {}))

        # active learning (BUS-space): refine on the full population. The acquisition
        # recomputes its own (logl_hat, h_i) target from p0 each AL iteration, so it
        # tracks the surrogate as it improves. Afterwards the population log-likelihoods
        # are re-evaluated with the REFINED surrogate (surrogate-only, mean path — no
        # true-model calls) and the scaling constant is re-maxed, so the level below
        # (geval -> h_i -> seeds) and aCS all run against the same refined model.
        if active_learning and retrain_srg_model is not None and is_bus:
            retrain_srg_model(l_class=l_class,
                              X_candidate_points=u2x(u_j[0:-1,:].T),
                              p_samples=sp.stats.norm.cdf(u_j[-1,:]),
                              c=logl_hat,
                              p0=p0,
                              **(al_kwargs or {}))
            leval = l_class.log_likelihood(u2x(u_j[0:-1,:].T), srg_prediction_model).flatten()
            cur_predictions = l_class.last_prediction.flatten()
            logl_hat = max(logl_hat, max(leval))   # re-max: keeps the BUS constraint c*L <= 1

        # compute the limit state function (Ref.1 Eq.12)
        geval = h_LSF(u_j.T[:,-1], logl_hat, leval)   # evaluate LSF (Ref.1 Eq.12)
        # nan arises when log(cdf(pi))=-inf and log_L=-inf combine as -inf+inf;
        # these samples have negligible prior AND negligible likelihood → firmly in safe region
        geval = np.where(np.isnan(geval), np.inf, geval)

        # sort values in ascending order
        idx = np.argsort(geval)
        predictions[i] = cur_predictions[idx].flatten()

        # order the samples according to idx
        u_j_sort = u_j[:,idx]
        samplesU['total'].append(u_j_sort.T)   # store the ordered samples

        # intermediate level
        h[i] = np.percentile(geval, p0*100, interpolation='midpoint')

        # number of failure points in the next level
        nF = int(sum(geval <= max(h[i],0)))

        #check if h[i] is nan
        if np.isnan(h[i]):
            print("h[i] is nan")
            print(f"leval: {leval}")
            print(f"logl_hat: {logl_hat}")
            print(f"h[i]: {h[i]}")
            print(f"geval: {geval}")
            print(f"p0: {p0}")
            print(f"nF: {nF}")
            print(f"N: {N}")

        # assign conditional probability to the level
        if h[i] < 0:
            h[i]      = 0
            prob[i-1] = nF/N
        else:
            prob[i-1] = p0
        print('\n-Threshold level ', i, ' = ', h[i])

        # select seeds and randomize the ordering (to avoid bias)
        seeds     = u_j_sort[:,:nF]
        idx_rnd   = np.random.permutation(nF)
        rnd_seeds = seeds[:,idx_rnd]            # non-ordered seeds
        samplesU['seeds'].append(seeds.T)         # store seeds

        # sampling process using adaptive conditional sampling
        u_j, leval, lam, sigma, accrate, cur_predictions = AL_aCS_aBUS(N, lam, h[i], rnd_seeds, logl_hat, h_LSF, l_class, srg_prediction_model, u2x)

        print('\t*aCS lambda =', lam, '\t*aCS sigma =', sigma[0], '\t*aCS accrate =', accrate)

        # update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
        l_new    = max(logl_hat, max(leval))
        h[i]     = h[i] - logl_hat + l_new
        logl_hat = l_new
        print(' Modified threshold level ', i, '= ', h[i])

        # decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
        p = np.random.uniform(low=np.zeros(N),
                              high=np.min([np.ones(N), np.exp(leval - logl_hat + h[i])], axis=0))
        u_j[-1,:] = sp.stats.norm.ppf(p)   # to the standard space
        i += 1
        if h[i-1] <= 0:
            break
    # number of intermediate levels
    

    # store final posterior samples
    samplesU['total'].append(u_j.T)  # store final failure samples (non-ordered)
    predictions[i] = cur_predictions

    # delete unnecesary data
    m = i + 1
    if m <= max_it:
        predictions = predictions[:m]
        prob = prob[:m]
        h    = h[:m]


    # transform the samples to the physical (original) space
    for i in range(m):
        pp = sp.stats.norm.cdf(samplesU['total'][i][:,-1])
        samplesX.append(np.concatenate((u2x(samplesU['total'][i][:,:-1]), pp.reshape(-1,1)), axis=1))

    # return h, samplesU, samplesX, logcE, c, sigma, predictions
    return samplesX, predictions[-1]