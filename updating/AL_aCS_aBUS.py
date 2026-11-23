import numpy as np
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
adaptive conditional sampling algorithm for aBUS
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Matthias Willer (matthias.willer@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-05
---------------------------------------------------------------------------
Modified by:
Dafydd Cotoarbă (TUM Georg Nemetschek Institute and Engineering Risk Analysis Group, TUM), 2025-2026
* likelihood evaluated through the surrogate prediction model
* all chains advanced in parallel, one step at a time; the proposal scale is adapted
  after about every 100 candidates (Ref. 1, Eqs. 25-26), as in the original every Na chains
* the current state of each chain is not re-evaluated; its log-likelihood and prediction
  are carried over from the previous step
---------------------------------------------------------------------------
Input:
* N         : number of samples to be generated
* lam       : scaling parameter lambda
* h         : current intermediate level
* u_j       : seeds used to generate the new samples
* log_L_fun : log-likelihood function
* l         : =-log(c) ~ scaling constant of BUS for the current level
* gl        : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jp1      : next level samples
* leval      : log-likelihood function of the new samples
* new_lambda : next scaling parameter lambda
* sigma      : spread of the proposal
* accrate    : acceptance rate of the samples
---------------------------------------------------------------------------
NOTES
* The way the initial standard deviation is computed can be changed in line 69.
By default we use option 'a' (it is equal to one).
In option 'b', it is computed from the seeds.
* The final accrate might differ from the target 0.44 when no adaptation is 
performed. Since at the last level almost all seeds are already in the failure 
domain, only a few are selected to complete the required N samples. The final 
accrate is computed for those few samples
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

def AL_aCS_aBUS(N, lambd_old, tau, theta_seeds, logl_hat, h_LSF, l_class, srg_model, u2x):
    n, Ns = theta_seeds.shape   # dimension and number of seeds
    
    # number of samples per chain
    Nchain = np.ones(Ns,dtype=int)*int(np.floor(N/Ns))
    Nchain[:np.mod(N,Ns)] = Nchain[:np.mod(N,Ns)]+1

    # initialization
    theta_chain = np.zeros((n,N))         # generated samples 
    leval       = np.zeros(N)             # store lsf evaluations
    predictions = np.zeros(N)
    acc         = np.zeros(N,dtype=int)   # store acceptance

    # adaptation of the proposal scale (Ref. 1, Sec. 4.2, Eqs. 25-26). The original
    # code runs the chains one after another and adapts every Na chains, i.e. about
    # every 100 samples. Here all chains advance together, one step at a time, so the
    # scale is adapted as soon as about 100 candidates have been proposed since the
    # last adaptation, and once more at the end of the level for the next level.
    star_a  = 0.44   # target acceptance rate
    n_adapt = 100    # candidates between adaptations

    # 1. compute the standard deviation
    opc = 'a'
    if opc == 'a': # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == 'b': # 1b. sigma = sigma_hat (sample standard deviations)
        sigma_0 = np.std(theta_seeds, axis=1)
    else:
        raise RuntimeError('Choose a or b')

    # 2. iteration
    lambd = lambd_old   # scaling parameter \in (0,1)

    # a. compute correlation parameter
    i     = 0                                      # index of the adaptation within this level
    sigma = np.minimum(lambd*sigma_0, np.ones(n))  # Ref. 1 Eq. 23
    rho   = np.sqrt(1-sigma**2)                    # Ref. 1 Eq. 24
    n_acc, n_prop = 0, 0                           # accepted / proposed since the last adaptation

    def adapt(lambd, i, n_acc, n_prop):
        hat_acc = n_acc/n_prop                                      # Ref. 1 Eq. 25
        zeta    = 1/np.sqrt(i+1)
        lambd   = np.exp(np.log(lambd) + zeta*(hat_acc-star_a))     # Ref. 1 Eq. 26
        sigma   = np.minimum(lambd*sigma_0, np.ones(n))
        return lambd, sigma, np.sqrt(1-sigma**2)

    max_nr_samples = max(Nchain)

    cur_mask = Nchain >= 1
    nr_idx = np.sum(cur_mask)

    # initialize with current seed
    theta_chain[:,0:nr_idx] = theta_seeds[:,cur_mask]   
    leval[0:nr_idx] = l_class.log_likelihood(u2x(theta_chain[0:-1,0:nr_idx].T), srg_model).flatten()

    predictions[0:nr_idx] = l_class.last_prediction.flatten()
    acc[0:nr_idx] = 0

    previous_start = 0
    previous_idx = nr_idx


    for sample_nr in range(2,max_nr_samples+1):
        cur_mask = Nchain >= sample_nr
        nr_idx = np.sum(cur_mask) + previous_idx

        # start from the current chain states; their log-likelihoods and predictions
        # are already known from the previous step, so they are copied, not re-evaluated
        theta_chain[:,previous_idx:nr_idx] = theta_chain[:,previous_start:previous_idx][:,cur_mask]
        leval[previous_idx:nr_idx] = leval[previous_start:previous_idx][cur_mask]
        predictions[previous_idx:nr_idx] = predictions[previous_start:previous_idx][cur_mask]
        acc[previous_idx:nr_idx] = 0

        # generate candidate sample, centered on the CURRENT chain state (the
        # batch just initialized at [previous_idx, nr_idx) from the previous step;
        # Ref. 1 Eq. 22). Indexing 0..batch-1 here would re-center every proposal
        # on the seeds, breaking the Markov chain.
        v_star = np.zeros((n, nr_idx-previous_idx))
        for cur_idx in range(nr_idx-previous_idx):
            v_star[:,cur_idx] = np.random.normal(loc=rho*theta_chain[:,previous_idx+cur_idx].reshape(1,-1), scale=sigma)

        leval_star = l_class.log_likelihood(u2x(v_star[0:-1,:].T), srg_model).flatten()
        heval_star = h_LSF(v_star.T[:,-1], logl_hat, leval_star)

        # Acceptance mask
        mask_leval_star = heval_star <= tau

        # Change accepted samples
        theta_chain[:,previous_idx:nr_idx][:,mask_leval_star] = v_star[:,mask_leval_star]
        leval[previous_idx:nr_idx][mask_leval_star] = leval_star[mask_leval_star]
        acc[previous_idx:nr_idx][mask_leval_star] = 1
        predictions[previous_idx:nr_idx][mask_leval_star] = l_class.last_prediction.flatten()[mask_leval_star]

        # adapt the proposal scale once enough candidates have been proposed
        n_acc  += int(np.sum(mask_leval_star))
        n_prop += nr_idx - previous_idx
        if n_prop >= n_adapt:
            lambd, sigma, rho = adapt(lambd, i, n_acc, n_prop)
            i += 1
            n_acc, n_prop = 0, 0

        # Update previous index
        previous_start = previous_idx
        previous_idx = nr_idx

    # next level lambda (adapted with the remaining candidates, if any)
    if n_prop > 0:
        lambd, sigma, rho = adapt(lambd, i, n_acc, n_prop)
    new_lambda = lambd

    # compute mean acceptance rate of all chains
    accrate = np.sum(acc)/(N-Ns)

    return theta_chain, leval, new_lambda, sigma, accrate, predictions