import torch
from torch.distributions import Normal
import numpy as np
from loss.eig import EIGStepLoss
from attrdictionary import AttrDict


@torch.no_grad()
def get_traces(model, experiment, T=30, batch_size=40, time_token=False):
    """ Get history

    Args:
        T (int): number of proposed designs in a trajectory
        batch_size (int): number of rollouts
        time_token (bool): whether to use time token
    """
    model.eval()

    theta_shape = experiment.sample_theta((batch_size)).shape
    
    # sample a batch of normalised data
    batch = experiment.sample_batch(batch_size)

    for t in range(T):
        if time_token:
            batch.t = torch.tensor([(T-t)/T])

        idx = model.forward(batch).design_out.idx      # [B, 1]

        batch = experiment.update_batch(batch, idx)

    # collect theta0
    theta_0 = batch.target_theta.reshape(*theta_shape)      # [B, (K, )D]

    # collect history
    x = experiment.unnormalise_design(batch.context_x)      # [B, T, D_x]
    y = batch.context_y                                     # [B, T, D_y]

    return theta_0, x, y


@torch.no_grad()
def compute_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), batch_size=40, stepwise=False):
    """ Evaluate the lower and upper bounds of EIG from a minibatch of the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    criterion = EIGStepLoss(L, batch_size, experiment.log_likelihood, reduction='none')

    pce_losses = []
    nmc_losses = []

    thetas = experiment.sample_theta((L, batch_size))
    
    thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)          # [L+1, B, (K, )D]

    if stepwise:
        for t in range(T):
            pce_loss, nmc_loss = criterion(y[:, t], x[:, t], thetas)  # [B]
            pce_losses.append(pce_loss)
            nmc_losses.append(nmc_loss)
                
        pce_losses = torch.stack(pce_losses, dim=-1)  # [B, T]
        nmc_losses = torch.stack(nmc_losses, dim=-1)  # [B, T]
    else:
        for t in range(T):
            pce_losses, nmc_losses = criterion(y[:, t], x[:, t], thetas)  # [B]

    # Calculate bounds
    pce_losses = torch.log(torch.tensor(L + 1)) - pce_losses  # [B(, T)]
    nmc_losses = torch.log(torch.tensor(L)) - nmc_losses      # [B(, T)]  

    return pce_losses, nmc_losses


@torch.no_grad()
def eval_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), M=2000, batch_size=40, stepwise=False, err_type='se'):
    """ Evaluate the lower and upper bounds of EIG from the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    max_step = (M + batch_size - 1) // batch_size

    pce_list = []
    nmc_list = []

    for step in range(max_step):
        start_idx = step * batch_size
        end_idx = min((step + 1) * batch_size, M)
        pce_loss, nmc_loss = compute_EIG_from_history(experiment, theta_0, x[start_idx:end_idx], y[start_idx:end_idx], L, end_idx - start_idx, stepwise)
        
        pce_list.append(pce_loss)
        nmc_list.append(nmc_loss)

    # Stack bounds
    pce = torch.cat(pce_list, dim=0)   # [M(, T)]
    nmc = torch.cat(nmc_list, dim=0)   # [M(, T)]

    # Calculate mean and std
    M = pce.shape[0]
    pce_mean = torch.mean(pce, dim=0)    # [T]
    pce_err = torch.std(pce, dim=0)     # [T]
    nmc_mean = torch.mean(nmc, dim=0)    # [T]
    nmc_err = torch.std(nmc, dim=0) / np.sqrt(M)     # [T]

    # Error type
    if err_type == 'se':
        pce_err = pce_err / np.sqrt(M)
        nmc_err = nmc_err / np.sqrt(M)
    elif err_type == 'ci':
        pce_err = 1.96 * pce_err / np.sqrt(M)
        nmc_err = 1.96 * nmc_err / np.sqrt(M)
    elif err_type == 'std':
        pass
    else:
        raise ValueError(f"Unknown err_type: {err_type}")

    pce_mean = pce_mean.cpu()
    pce_err = pce_err.cpu()
    nmc_mean = nmc_mean.cpu()
    nmc_err = nmc_err.cpu()

    bounds = AttrDict(pce_mean=pce_mean, pce_err=pce_err, nmc_mean=nmc_mean, nmc_err=nmc_err)

    return bounds

@torch.no_grad()
def eval_boed(model, experiment, T=30, L=int(1e6), M=2000, batch_size=40, time_token=False, stepwise=False, err_type='se'):
    """ Final evaluation of the EIG bounds for ALINE

    Args:
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        M (int): number of outer samples
        batch_size (int): mini batch size of outer samples
        time_token (bool): whether to use time token
    """
    model.eval()

    max_step = (M + batch_size - 1) // batch_size

    pce_list = []
    nmc_list = []

    for step in range(max_step):
        theta_0, x, y = get_traces(model, experiment, T, batch_size, time_token)
        pce, nmc = compute_EIG_from_history(experiment, theta_0, x, y, L, batch_size, stepwise)
        pce_list.append(pce)
        nmc_list.append(nmc)
        print(f"Step {step}: PCE {pce.mean(dim=0)}, NMC {nmc.mean(dim=0)}")

    # Stack bounds
    pce = torch.cat(pce_list, dim=0)   # [M(, T)]
    nmc = torch.cat(nmc_list, dim=0)   # [M(, T)]

    # Calculate mean and std
    M = pce.shape[0]
    pce_mean = torch.mean(pce, dim=0)    # [T]
    pce_err = torch.std(pce, dim=0)     # [T]
    nmc_mean = torch.mean(nmc, dim=0)    # [T]
    nmc_err = torch.std(nmc, dim=0)     # [T]

    # Error type
    if err_type == 'se':
        pce_err = pce_err / np.sqrt(M)
        nmc_err = nmc_err / np.sqrt(M)
    elif err_type == 'ci':
        pce_err = 1.96 * pce_err / np.sqrt(M)
        nmc_err = 1.96 * nmc_err / np.sqrt(M)
    elif err_type == 'std':
        pass
    else:
        raise ValueError(f"Unknown err_type: {err_type}")


    pce_mean = pce_mean.cpu()
    pce_err = pce_err.cpu()
    nmc_mean = nmc_mean.cpu()
    nmc_err = nmc_err.cpu()

    bounds = AttrDict(pce_mean=pce_mean, pce_err=pce_err, nmc_mean=nmc_mean, nmc_err=nmc_err)

    return bounds

def compute_ll(
    value: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Computes coordinate-wise log-likelihood for a Gaussian mixture model.

    Args:
        value:   [B, n_target]
        means:   [B, n_target, C]
        stds:    [B, n_target, C]
        weights: [B, n_target, C]

    Returns:
        log_likelihood: [B, n_target]
    """
    if value.ndim == means.ndim - 1:
        value = value.unsqueeze(-1)

    components = Normal(means, stds, validate_args=False)
    log_probs = components.log_prob(value)

    weighted_log_probs = log_probs + torch.log(weights.clamp_min(1e-12))
    return torch.logsumexp(weighted_log_probs, dim=-1)


def compute_rmse(target_values, mixture_means, mixture_stds, mixture_weights):
    # TODO: you have to apply the target mask to the target_values and mixture components first, need to update later.
    """
    Compute RMSE between target values and predictions from Gaussian mixture model

    Args:
        target_values: Ground truth values [batch_size, n_targets, dim_y]
        mixture_means: Means of mixture components [batch_size, n_targets, n_components]
        mixture_stds: Standard deviations of mixture components [batch_size, n_targets, n_components]
        mixture_weights: Weights of mixture components [batch_size, n_targets, n_components]

    Returns:
        rmse: RMSE values [batch_size, n_targets]
    """
    # Calculate weighted mean for each target point
    weighted_means = torch.sum(mixture_weights * mixture_means, dim=-1)  # [batch_size, n_targets]

    # Calculate squared error
    squared_error = (target_values.squeeze(-1) - weighted_means) ** 2

    # Calculate RMSE
    rmse = torch.sqrt(torch.mean(squared_error, dim=-1))  # [batch_size]

    return rmse





# -------------------------- Simple Pendulum ---------------------# 

import torch
import numpy as np

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sample_theta(experiment, shape):
    """
    Compatible avec un sample_theta qui accepte seulement un int.
    shape peut être int, tuple ou list.
    """
    if isinstance(shape, int):
        return experiment.sample_theta(shape)

    shape = tuple(shape)
    n = int(np.prod(shape))
    theta = experiment.sample_theta(n)
    return theta.reshape(*shape, experiment.dim_theta)


def _initial_y(experiment, B, device):
    return experiment.simulator.get_initial_observation(B).to(device)


# ---------------------------------------------------------------------
# Rollout ALINE pour SimplePendulum
# ---------------------------------------------------------------------

@torch.no_grad()
def get_traces_pendulum(model, experiment, T=30, batch_size=40, time_token=False):
    """
    Simule T étapes avec le modèle ALINE et retourne :
        theta_0: [B, dim_theta]
        x:       [B, T, dim_x]
        y:       [B, T, dim_y]

    Important :
    - on enlève le contexte initial dummy.
    - on n'appelle pas experiment.unnormalise_design.
    """

    model.eval()
    device = experiment.device if hasattr(experiment, "device") else next(model.parameters()).device

    batch = experiment.sample_batch(batch_size)

    for t in range(T):
        if time_token:
            batch.t = torch.tensor([(T - t) / T], device=device)

        pred = model.forward(batch)
        idx = pred.design_out.idx

        batch = experiment.update_batch(batch, idx)

    theta_0 = batch.theta  # [B, dim_theta]

    # Remove initial dummy context.
    n0 = getattr(experiment, "n_context_init", 1)
    x = batch.context_x[:, n0:]  # [B, T, dim_x]
    y = batch.context_y[:, n0:]  # [B, T, dim_y]

    return theta_0, x, y



# ---------------------------------------------------------------------
# Sequential likelihood for SimplePendulum
# ---------------------------------------------------------------------

@torch.no_grad()
def history_logp_pendulum(experiment, theta, x, y, stepwise=False):
    """
    Computes log p(y_{1:T} | theta, x_{1:T}) for SimplePendulum.

    Args:
        theta:
            [B, dim_theta] or [S, B, dim_theta]
        x:
            [B, T, dim_x]
        y:
            [B, T, dim_y]
        stepwise:
            if False: returns cumulative logp over full trajectory
            if True: returns cumulative logp at each time step

    Returns:
        if theta is [B, D]:
            [B] if not stepwise else [B, T]
        if theta is [S, B, D]:
            [S, B] if not stepwise else [S, B, T]
    """

    device = x.device
    B, T, dim_x = x.shape
    dim_y = y.shape[-1]

    theta = theta.to(device)
    x = x.to(device)
    y = y.to(device)

    if theta.ndim == 2:
        # theta: [B, dim_theta]
        y_prev = _initial_y(experiment, B, device)

        logps = []

        for t in range(T):
            logp_t = experiment.simulator._outcome_logp(
                y=y[:, t],
                theta=theta,
                design=x[:, t],
                last_y=y_prev,
            )  # [B]

            logps.append(logp_t)
            y_prev = y[:, t]

        logps = torch.stack(logps, dim=-1)  # [B, T]
        cum_logps = logps.cumsum(dim=-1)

        return cum_logps if stepwise else cum_logps[:, -1]

    elif theta.ndim == 3:
        # theta: [S, B, dim_theta]
        S = theta.shape[0]

        y_prev = _initial_y(experiment, B, device)
        y_prev = y_prev.unsqueeze(0).expand(S, B, dim_y)

        logps = []

        for t in range(T):
            y_t = y[:, t].unsqueeze(0).expand(S, B, dim_y)
            x_t = x[:, t].unsqueeze(0).expand(S, B, dim_x)

            logp_t = experiment.simulator._outcome_logp(
                y=y_t,
                theta=theta,
                design=x_t,
                last_y=y_prev,
            )  # [S, B]

            logps.append(logp_t)
            y_prev = y_t

        logps = torch.stack(logps, dim=-1)  # [S, B, T]
        cum_logps = logps.cumsum(dim=-1)

        return cum_logps if stepwise else cum_logps[:, :, -1]

    else:
        raise ValueError(f"Unexpected theta shape: {theta.shape}")
    

# ---------------------------------------------------------------------
# EIG bounds for SimplePendulum
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_EIG_from_history_pendulum(
    experiment,
    theta_0,
    x,
    y,
    L=10000,
    batch_size=None,
    stepwise=False,
    L_chunk=2048,
):
    """
    PCE / NMC bounds for SimplePendulum.

    Args:
        theta_0: [B, dim_theta]
        x:       [B, T, dim_x]
        y:       [B, T, dim_y]
        L:       number of contrastive theta samples
        stepwise:
            False -> returns [B]
            True  -> returns [B, T]

    Returns:
        pce: [B] or [B, T]
        nmc: [B] or [B, T]
    """

    device = x.device
    theta_0 = theta_0.to(device)
    x = x.to(device)
    y = y.to(device)

    B = x.shape[0]

    # True likelihood log p(y | theta_0, x)
    logp_true = history_logp_pendulum(
        experiment,
        theta_0,
        x,
        y,
        stepwise=stepwise,
    )  # [B] or [B, T]

    # Streaming logsumexp over contrastive samples.
    contrastive_lse = None
    n_done = 0

    while n_done < L:
        l_now = min(L_chunk, L - n_done)

        theta_c = _sample_theta(experiment, (l_now, B)).to(device)
        # theta_c: [l_now, B, dim_theta]

        logp_c = history_logp_pendulum(
            experiment,
            theta_c,
            x,
            y,
            stepwise=stepwise,
        )  # [l_now, B] or [l_now, B, T]

        lse_c = torch.logsumexp(logp_c, dim=0)  # [B] or [B, T]

        if contrastive_lse is None:
            contrastive_lse = lse_c
        else:
            contrastive_lse = torch.logaddexp(contrastive_lse, lse_c)

        n_done += l_now

    # PCE denominator includes theta_0 + L contrastives.
    pce_den_lse = torch.logaddexp(logp_true, contrastive_lse)

    log_L_plus_1 = torch.log(torch.tensor(float(L + 1), device=device))
    log_L = torch.log(torch.tensor(float(L), device=device))

    pce = log_L_plus_1 + logp_true - pce_den_lse

    # NMC denominator uses only contrastive samples.
    nmc = log_L + logp_true - contrastive_lse

    return pce, nmc


@torch.no_grad()
def eval_boed_pendulum(
    model,
    experiment,
    T=30,
    L=10000,
    M=2000,
    batch_size=40,
    time_token=False,
    stepwise=False,
    err_type="se",
    L_chunk=2048,
):
    """
    Replacement for eval_boed, compatible with SimplePendulum.
    """

    model.eval()

    pce_list = []
    nmc_list = []

    max_step = (M + batch_size - 1) // batch_size

    for step in range(max_step):
        b_now = min(batch_size, M - step * batch_size)

        theta_0, x, y = get_traces_pendulum(
            model=model,
            experiment=experiment,
            T=T,
            batch_size=b_now,
            time_token=time_token,
        )

        pce, nmc = compute_EIG_from_history_pendulum(
            experiment=experiment,
            theta_0=theta_0,
            x=x,
            y=y,
            L=L,
            batch_size=b_now,
            stepwise=stepwise,
            L_chunk=L_chunk,
        )

        pce_list.append(pce.detach().cpu())
        nmc_list.append(nmc.detach().cpu())

        print(
            f"Step {step + 1}/{max_step}: "
            f"PCE={pce.mean(dim=0).detach().cpu()}, "
            f"NMC={nmc.mean(dim=0).detach().cpu()}"
        )

    pce = torch.cat(pce_list, dim=0)  # [M] or [M, T]
    nmc = torch.cat(nmc_list, dim=0)  # [M] or [M, T]

    M_eff = pce.shape[0]

    pce_mean = pce.mean(dim=0)
    nmc_mean = nmc.mean(dim=0)

    pce_std = pce.std(dim=0, unbiased=True)
    nmc_std = nmc.std(dim=0, unbiased=True)

    if err_type == "se":
        pce_err = pce_std / np.sqrt(M_eff)
        nmc_err = nmc_std / np.sqrt(M_eff)
    elif err_type == "ci":
        pce_err = 1.96 * pce_std / np.sqrt(M_eff)
        nmc_err = 1.96 * nmc_std / np.sqrt(M_eff)
    elif err_type == "std":
        pce_err = pce_std
        nmc_err = nmc_std
    else:
        raise ValueError(f"Unknown err_type: {err_type}")

    bounds = AttrDict()
    bounds.pce_mean = pce_mean
    bounds.pce_err = pce_err
    bounds.nmc_mean = nmc_mean
    bounds.nmc_err = nmc_err

    return bounds

