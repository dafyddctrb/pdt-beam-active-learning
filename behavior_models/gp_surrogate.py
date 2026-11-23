import os
from pathlib import Path
import numpy as np
import torch
import gpytorch
from datetime import datetime
import matplotlib.pyplot as plt
import gpytorch.settings
from linear_operator.utils.errors import NotPSDError
from sklearn.preprocessing import StandardScaler


def bus_recompute_level(g_base, ln_p, logl_hat, p0):
    """Recompute the BUS scaling constant and SuS level from the CURRENT surrogate.

    The frozen (logl_hat, h_i) pair handed down by the SuS loop goes stale while the
    surrogate is refined inside the AL loop: both were derived from the
    pre-refinement leval. This recomputes them self-consistently each iteration:

        mu_logL      = ln_p - g_base            # g_base = the logL LSF at logl_hat=0, b=0
        logl_hat_cur = max(logl_hat, max(mu_logL))   # re-max: keeps BUS c*L <= 1
        geval        = g_base + logl_hat_cur         # h_LSF under current surrogate
        h_i          = max(p0-percentile(geval), 0)  # clamped like the SuS driver

    As the surrogate converges the percentile drops and h_i -> 0: the moving
    acquisition target lands on the true posterior boundary (b = 0).

    Returns
    -------
    (logl_hat_cur, h_i, geval) : (float, float, np.ndarray)
    """
    g_base = np.asarray(g_base, dtype=float)
    mu_logL = np.asarray(ln_p, dtype=float) - g_base
    logl_hat_cur = max(float(logl_hat), float(np.max(mu_logL)))
    geval = g_base + logl_hat_cur
    h_i = max(float(np.percentile(geval, p0 * 100, method='midpoint')), 0.0)
    return logl_hat_cur, h_i, geval


def reanchored_delta_and_ref(ln_p, mu, logl_hat_now, h_now, geval_now, ref):
    """Re-anchored common-reference delta for AL stopping (seeded_delta method).

    Holds the PREVIOUS iteration's level (`ref`) fixed and measures how much the
    CURRENT surrogate's log-likelihood `mu` moves the BUS failure fraction at that
    level, then returns the re-anchored reference for the next iteration. Using one
    frozen level for both sides isolates the surrogate change (not the moving
    level); re-anchoring each call lets the level walk toward the truth.

        ref : None (cold start), or a reference tuple
              (logl_hat, h, pf_below)  where pf_below is mean(geval <= h)
              captured when that reference was frozen.
        returns (delta, new_ref)   # delta is None on cold start

    The reference persists across retrain_model calls, so at iteration 0 the delta
    is the SEEDED single-look "did the estimate move since the last call".
    """
    pf_before_now = float(np.mean(np.asarray(geval_now, dtype=float) <= h_now))
    new_ref = (float(logl_hat_now), float(h_now), pf_before_now)
    if ref is None:
        return None, new_ref
    ll_ref, h_ref, pf_below_ref = ref
    ln_p = np.asarray(ln_p, dtype=float)
    mu = np.asarray(mu, dtype=float)
    pf_after = float(np.mean(((ln_p - mu) + ll_ref) <= h_ref))
    return abs(pf_after - pf_below_ref), new_ref


def beta_reltol_from_pf_reltol(rel_pf, Pf):
    """Convert a RELATIVE-Pf tolerance into the equivalent RELATIVE-beta tolerance at
    operating point Pf, so a beta-space stopping test (beta = -Phi^-1(Pf)) is held to
    the SAME effective strictness as the Pf-space one. This lets all thresholds be
    specified once in Pf space; the beta_* criteria auto-convert at the live Pf.

    Since Pf = Phi(-beta), a small perturbation maps as
        d(beta)/beta = (d Pf / Pf) * Phi(-beta) / (phi(beta) * beta)
    so rel_beta = rel_pf * Phi(-beta) / (phi(beta) * beta). For Pf >= 0.5 (beta <= 0)
    the map is singular -> fall back to the identity (no conversion).
    """
    from scipy.stats import norm
    Pf = min(max(float(Pf), 1e-12), 1.0 - 1e-12)
    b = -float(norm.ppf(Pf))
    if b <= 1e-9:
        return float(rel_pf)
    return float(rel_pf) * float(norm.cdf(-b)) / (float(norm.pdf(b)) * b)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.RBFKernel()
        base_kernel.lengthscale = 1.5
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class DependentGPRModels:
    
    def __init__(self):
        torch.set_num_threads(1)
        self.model = None
        self.likelihood = None
        self.scaler_x = None
        self.scaler_y = None

    def train(self, x_train, y_train, scaler_x, scaler_y, n_epochs=50, lr=0.01,
              noise=False, init_state_dict: dict = None):
        """Fit an exact GP (constant mean, scaled RBF kernel) to standardised data.
        `init_state_dict` warm-starts the hyperparameters from a previous fit. Without
        `noise`, the likelihood noise is fixed at 1e-3 (the model is deterministic)."""
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        self.scaler_x = scaler_x.fit(x_train.numpy())
        x_scaled = torch.tensor(self.scaler_x.transform(x_train.numpy()), dtype=torch.float32)
        self.scaler_y = scaler_y.fit(y_train.numpy().reshape(-1, 1))
        y_scaled = torch.tensor(self.scaler_y.transform(y_train.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if not noise:
            self.likelihood.noise = 1e-3
            self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.model = ExactGPModel(x_scaled, y_scaled, self.likelihood)

        if init_state_dict is not None:
            try:
                self.model.load_state_dict(init_state_dict['model'], strict=False)
                self.likelihood.load_state_dict(init_state_dict['likelihood'], strict=False)
            except Exception:
                pass  # fall back to random init

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        losses = []
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = -mll(self.model(x_scaled), y_scaled)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.model.eval()
        self.likelihood.eval()
        return np.array(losses), self.model.state_dict()

    def predict(self, x, return_std: bool = True):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        X_scaled = torch.tensor(self.scaler_x.transform(x.numpy()), dtype=torch.float32)

        if not return_std:
            # Mean-only: skip the predictive covariance entirely. For an exact GP the
            # mean (k_*^T alpha) is independent of the variance solve, so this is
            # bit-identical to the full path — it just avoids the expensive per-query
            # Cholesky solves. Use wherever the caller discards the std (the Gaussian
            # likelihood evaluations over every SuS/aCS sample). Second slot is None.
            with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                mean = self.likelihood(self.model(X_scaled)).mean
                if mean.dim() == 1:
                    mean = mean.reshape(-1, 1)
                mean_unscaled = self.scaler_y.inverse_transform(mean.detach().cpu().numpy())
            return np.array(mean_unscaled), None

        with torch.no_grad():
            model_output = self.model(X_scaled)
            observed_pred = self.likelihood(model_output)
                
            mean = observed_pred.mean
            if mean.dim() == 1:
                mean = mean.reshape(-1, 1)
            stddev = observed_pred.stddev
            if stddev.dim() == 1:
                stddev = stddev.reshape(-1, 1)

            
            # Rescale predictions back to original scale
            mean_np = mean.detach().cpu().numpy()
            std_np = stddev.detach().cpu().numpy()
            mean_unscaled = self.scaler_y.inverse_transform(mean_np)

            if isinstance(self.scaler_y, StandardScaler):
                scale = np.asarray(self.scaler_y.scale_, dtype=std_np.dtype)
                stddev_unscaled = std_np * scale
            else:
                stddev_unscaled = (
                    self.scaler_y.inverse_transform(mean_np + std_np) - mean_unscaled
                )
        return np.array(mean_unscaled), np.array(stddev_unscaled)


class SRGModel:

    def __init__(self, get_true_response_function: callable, plot_dir: Path = None):
        self.cur_model = DependentGPRModels()
        self.X_train_current = None
        self.Y_train_current = None
        self.likelihood_calls = 0
        self.get_true_response_function = get_true_response_function
        self.plot_dir = plot_dir

    def predict(self, x, return_std: bool = True):
        return self.cur_model.predict(x, return_std=return_std)
    
    def train_model(self, X_train_current, Y_train_current=None,
                    noise: bool = False, lr: float = 1e-2, epochs: int = 2_00,
                    plot_path: Path = "t"):
        """
        Train the model with new samples.
        """
        self.X_train_current = X_train_current
        if Y_train_current is None:
            Y_train_current = self.get_true_response_function(E=self.X_train_current[:, 0], I=self.X_train_current[:, 1])
            self.likelihood_calls += X_train_current.shape[0]
        self.Y_train_current = Y_train_current
        
        X_train_current_tensor = torch.tensor(X_train_current, dtype=torch.float32)
        Y_train_current_tensor = torch.tensor(Y_train_current, dtype=torch.float32)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        
        self.cur_model.train(
                    x_train=X_train_current_tensor, 
                    y_train=Y_train_current_tensor, 
                    scaler_x=scaler_x, 
                    scaler_y=scaler_y,
                    n_epochs=epochs,
                    lr=lr, 
                    noise=noise,
                )
        
        # # create parity plot with two subplots   
        if plot_path is not None and os.environ.get("PDT_NO_PLOTS") != "1":
            training_data_predicted_mean, _ = self.cur_model.predict(X_train_current_tensor)

            # create parity plot with two subplots   
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].scatter(Y_train_current, training_data_predicted_mean, label='Before')
            ax[0].set_xlabel('True Mean')
            ax[0].set_ylabel('Predicted Mean')
            ax[0].set_title('Training Data')
            ax[0].legend()
        #     # save figure
            if self.plot_dir is None:
                plt_dir = "plots/parity_plots"
            else:   
                plt_dir = f"{self.plot_dir}/parity_plots"
            # mkdir if not exists
            Path(plt_dir).mkdir(parents=True, exist_ok=True)
            time_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'{plt_dir}/train_test_parity_plot_{time_id}.png')
            plt.close()


    def retrain_model(self, l_class, prediction_model, X_candidate_params, X_candidate_slices,
                      bus_p_samples, bus_p0, bus_logl_hat=None,
                      acquisition_function="bus_u_learning", convergence_method="pf_stability",
                      convergence_criteria=0.05, min_u_threshold=2.0,
                      seeded_delta_eps=0.05,
                      n_iterations=10, min_train_points=15,
                      epochs=100, lr=1e-2):
        """BUS / log-likelihood-space active learning.

        Each iteration scores candidates by U = |g_lsf| / sigma_logL, where the
        log-likelihood moments (mu_logL, sigma_logL) come from l_class — the SAME
        likelihood the SuS loop uses — via
        ``log_likelihood(..., return_std=True)``.
        The BUS level (logl_hat, h_i) is recomputed from the current surrogate each
        iteration (self-consistent moving target, see bus_recompute_level), a true
        sample is taken at the min-U candidate, and the GP is refit (warm-started)
        until the chosen convergence criterion is met.

        Args:
            prediction_model: surrogate predictor (params, t_cur, return_std) -> (mean, std).
            X_candidate_params: (N, n_params) candidates in PARAMETER space.
            X_candidate_slices: (K, N, n_feat) surrogate-feature vectors per measurement
                time, used to place the new training point (slice -1 = current state).
            acquisition_function: 'bus_u_learning' (first order) or 'bus_u_learning_2nd'.
            convergence_method:
                'pf_stability' — relative change of Pf = P(g_lsf <= 0) across iterations
                    below convergence_criteria (needs >=1 added point to compare).
                'min_u' — the smallest U over candidates clears min_u_threshold
                    (every candidate confidently classified; AK-MCS, Echard 2011).
                'pf_bound' — the Pf bounds from the surrogate CI are tight:
                    (Pf+ - Pf-)/Pf <= convergence_criteria, with
                    Pf+ = P(geval - k*sigma <= 0), Pf- = P(geval + k*sigma <= 0).
                    Evaluable at a single iteration, so it can converge with NO point
                    added once min_train_points is met (Schobi/Sudret/Marelli 2017).
              Reliability-index (beta = -Phi^-1(Pf)) twins of the Pf criteria, more
              stable for small Pf (Moustapha, Marelli & Sudret 2022). Single-shot,
              mirroring pf_stability/pf_bound (the paper's consecutive-iteration guard
              was dropped so the only difference from the pf_* twin is the space):
                'beta_stability' — |b^(i) - b^(i-1)| / |b^(i)| <= criterion (β-space
                    twin of pf_stability; their BS without the 3-in-a-row guard).
                'beta_bound' — |b+ - b-| / |b| <= criterion, with b± from
                    geval -/+ 2*sigma (β-space twin of pf_bound; their BB w/o the guard).
                'combined' — both beta_stability AND beta_bound, 2 in a row (their Co;
                    retains the consecutive-iteration guard).
                'seeded_delta' — re-anchored common-reference delta plateau: freeze the
                    PREVIOUS level (logl_hat, h) and check whether the current surrogate
                    moves the failure fraction at it by less than seeded_delta_eps. The
                    reference persists across retrain calls, so iteration 0 is a seeded
                    cross-call look that can stop with ZERO points added when the
                    surrogate has not materially changed (isolates the surrogate from the
                    moving BUS level; unlike pf_stability which confounds the two).
            convergence_criteria: relative threshold for the Pf/beta methods, ALWAYS
                expressed in Pf-RELATIVE units. The pf_* tests use it directly; the
                beta_* tests auto-convert it to the equivalent beta-relative tolerance
                at the live operating Pf (beta_reltol_from_pf_reltol), so a beta_* and
                its pf_* twin are matched in strictness and differ only by the space.
            min_u_threshold: U threshold for 'min_u' (2.0 ~ Phi(-2) misclassification, AK-MCS).
            seeded_delta_eps: ABSOLUTE change in the BUS failure fraction for 'seeded_delta'
                (distinct from convergence_criteria, which is a relative Pf change).
            bus_p0: SuS conditional probability (the level is recomputed from it).
            min_train_points: convergence is refused below this many training points.
        """
        if acquisition_function not in ("bus_u_learning", "bus_u_learning_2nd"):
            raise ValueError(f"retrain_model is BUS-only; got {acquisition_function!r}")
        valid_conv = ("pf_stability", "min_u", "pf_bound", "beta_stability", "beta_bound",
                      "combined", "seeded_delta")
        if convergence_method not in valid_conv:
            raise ValueError(f"retrain_model convergence_method must be one of {valid_conv}; got {convergence_method!r}")
        second_order = acquisition_function == "bus_u_learning_2nd"
        from scipy.stats import norm as _norm
        _beta = lambda pf: float(-_norm.ppf(min(max(pf, 1e-12), 1.0 - 1e-12)))   # reliability index

        X = np.asarray(X_candidate_params, dtype=float)
        slices = np.asarray(X_candidate_slices, dtype=float)
        row = (X >= 0).all(axis=1) & (slices >= 0).all(axis=(0, 2))   # valid at every state
        X, slices = X[row], slices[:, row, :]
        ln_p = np.log(np.maximum(np.asarray(bus_p_samples).reshape(-1)[row], 1e-300))
        logl_hat = -np.inf if bus_logl_hat is None else float(bus_logl_hat)

        self.last_min_u_trace, Pf_prev, beta_prev, consec = [], None, None, 0
        # Re-anchored-delta reference (logl_hat, h, pf_before). PERSISTS across retrain
        # calls so iteration 0 has a seed from the previous call -> the seeded
        # single-look "do I need any point this step" decision (see seeded_delta).
        if not hasattr(self, "_reanchor_ref"):
            self._reanchor_ref = None
        print("### Retraining model... ###")
        it = 0
        rejected = []   # candidates whose refit failed; not proposed again
        for it in range(n_iterations):
            mu, sigma = l_class.log_likelihood(
                X, prediction_model, return_std=True, second_order=second_order)
            mu, sigma = mu.reshape(-1), sigma.reshape(-1)
            logl_hat, h_i, geval = bus_recompute_level(ln_p - mu, ln_p, logl_hat, bus_p0)
            U = np.abs(geval - h_i) / np.maximum(sigma, 1e-9)
            self.last_min_u_trace.append(float(U.min()))

            # Re-anchored common-reference delta (always computed; cheap). Holds the
            # PREVIOUS (logl_hat, h) fixed and measures how much the CURRENT surrogate's
            # mu moves the failure fraction at that level -> isolates the surrogate change
            # from the moving BUS level. At it==0 the delta is the SEEDED cross-call
            # signal; cold start (no seed) has _delta=None -> never converges -> enrich.
            _delta, self._reanchor_ref = reanchored_delta_and_ref(
                ln_p, mu, logl_hat, h_i, geval, self._reanchor_ref)

            Pf, min_u = float(np.mean(geval <= 0)), float(U.min())
            print(f"  AL iter {it}: min_u={min_u:.4f} | Pf={Pf:.4f}")

            # Single-iteration conditions shared by the beta_* / combined criteria.
            # convergence_criteria is a Pf-RELATIVE tolerance everywhere; the beta tests
            # auto-convert it to the equivalent beta-relative tolerance at the live Pf,
            # so beta_* and pf_* twins are held to matched strictness (only the space differs).
            beta = _beta(Pf)
            beta_tol = beta_reltol_from_pf_reltol(convergence_criteria, Pf)
            stab_ok = beta_prev is not None and abs(beta - beta_prev) / max(abs(beta), 1e-12) <= beta_tol
            beta_plus = _beta(float(np.mean((geval - 2.0 * sigma) <= 0)))    # geval - 2σ -> more failures
            beta_minus = _beta(float(np.mean((geval + 2.0 * sigma) <= 0)))   # geval + 2σ -> fewer failures
            bounds_ok = abs(beta_plus - beta_minus) / max(abs(beta), 1e-12) <= beta_tol

            if convergence_method == "min_u":
                converged = min_u >= min_u_threshold
            elif convergence_method == "pf_stability":
                converged = Pf_prev is not None and abs(Pf - Pf_prev) / max(Pf, 1e-15) < convergence_criteria
            elif convergence_method == "pf_bound":
                # Pf bounds from the surrogate CI: band can't move Pf past the criterion.
                Pf_plus = float(np.mean((geval - 1.96 * sigma) <= 0))
                Pf_minus = float(np.mean((geval + 1.96 * sigma) <= 0))
                converged = (Pf_plus - Pf_minus) / max(Pf, 1e-12) <= convergence_criteria
            elif convergence_method == "beta_stability":     # single-shot β-space twin of pf_stability
                converged = stab_ok
            elif convergence_method == "beta_bound":         # single-shot β-space twin of pf_bound
                converged = bounds_ok
            elif convergence_method == "seeded_delta":
                # Re-anchored delta plateau. At it==0 the delta is the SEEDED cross-call
                # signal -> a True here breaks before enrichment (zero-point step).
                # Cold start (no seed) has _delta=None -> never converges -> enrich.
                converged = _delta is not None and _delta < seeded_delta_eps
            else:                                             # 'combined' (Moustapha Co): 2 consecutive
                consec = consec + 1 if (stab_ok and bounds_ok) else 0
                converged = consec >= 2

            if converged and len(self.X_train_current) >= min_train_points:
                print(f"Converged at iteration {it} ({convergence_method})")
                break
            Pf_prev, beta_prev = Pf, beta

            # Sample the truth at the most informative candidate (current state) and refit.
            U_pick = np.array(U, dtype=float).ravel()
            U_pick[rejected] = np.inf
            j = int(np.argmin(U_pick))
            x_new = slices[-1, j].reshape(1, -1)
            y_new = self.get_true_response_function(E=x_new[:, 0], I=x_new[:, 1])
            self.likelihood_calls += 1
            X_try = np.concatenate([self.X_train_current, x_new])
            Y_try = np.concatenate([self.Y_train_current, y_new])
            prev_state = {'model': self.cur_model.model.state_dict(),
                          'likelihood': self.cur_model.likelihood.state_dict()}
            new_model = DependentGPRModels()
            try:
                new_model.train(x_train=torch.tensor(X_try, dtype=torch.float32),
                                y_train=torch.tensor(Y_try, dtype=torch.float32),
                                scaler_x=StandardScaler(), scaler_y=StandardScaler(),
                                n_epochs=epochs, lr=lr, init_state_dict=prev_state)
            except NotPSDError:
                # the true model was already evaluated, so the call still counts
                print(f"  [AL iter {it}] NotPSDError — skipping this candidate")
                rejected.append(j)
                continue
            self.X_train_current, self.Y_train_current, self.cur_model = X_try, Y_try, new_model
        print(f"### Retrained model after {it} iterations ###")
