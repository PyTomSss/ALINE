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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sample_theta(experiment, shape):
    if isinstance(shape, int):
        return experiment.sample_theta(shape)

    shape = tuple(shape)
    n = int(np.prod(shape))

    theta = experiment.sample_theta(n)
    dim_theta = theta.shape[-1]

    return theta.reshape(*shape, dim_theta)


def _initial_y(experiment, B, device):
    return experiment.simulator.get_initial_observation(B).to(device)


# ---------------------------------------------------------------------
# Rollout ALINE pour SimplePendulum
# ---------------------------------------------------------------------

def _move_batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def _get_dim_xi(experiment):
    """
    Returns the true simulator design dimension.
    Falls back to dim_x for old experiments.
    """
    if hasattr(experiment, "dim_xi"):
        return int(experiment.dim_xi)

    if hasattr(experiment, "simulator") and hasattr(experiment.simulator, "dim_xi"):
        return int(experiment.simulator.dim_xi)

    # Backward compatibility: old setup had dim_x == dim_xi.
    return int(experiment.dim_x)


def _extract_xi_from_x(x, experiment, xi_position="first"):
    """
    Extracts the physical control xi from the augmented ALINE input x.

    x:
        [..., dim_x]

    Returns:
        xi:
            [..., dim_xi]

    Convention:
        if xi_position == "first":
            x = concat([xi, previous_observation, ...])

        if xi_position == "last":
            x = concat([previous_observation, ..., xi])
    """

    dim_xi = _get_dim_xi(experiment)

    if x.shape[-1] == dim_xi:
        return x

    if x.shape[-1] < dim_xi:
        raise ValueError(
            f"Cannot extract dim_xi={dim_xi} from x.shape[-1]={x.shape[-1]}."
        )

    if xi_position == "first":
        return x[..., :dim_xi]

    elif xi_position == "last":
        return x[..., -dim_xi:]

    else:
        raise ValueError(f"Unknown xi_position={xi_position}.")
    

@torch.no_grad()
def get_traces_pendulum(
    model,
    experiment,
    T=30,
    batch_size=40,
    time_token=False,
    xi_position="first",
    return_augmented_x=True,
):
    """
    Simule T étapes avec le modèle ALINE.

    Returns
    -------
    theta_0:
        [B, dim_theta]

    x:
        [B, T, dim_x]
        Augmented ALINE inputs stored in context_x.

    xi:
        [B, T, dim_xi]
        True simulator controls extracted from x.

    y:
        [B, T, dim_y]

    Important:
    - x is what ALINE sees.
    - xi is what the simulator likelihood uses.
    """

    model.eval()
    device = experiment.device if hasattr(experiment, "device") else next(model.parameters()).device

    batch = experiment.sample_batch(batch_size)
    batch = _move_batch_to_device(batch, device)

    xs = []
    xis = []
    ys = []

    for t in range(T):
        if time_token:
            # Keep your convention. You may also use t / T if that is what training used.
            batch.t = torch.tensor([(T - t) / T], device=device)

        pred = model.forward(batch)
        idx = pred.design_out.idx

        idx_flat = idx.squeeze(-1) if idx.ndim > 1 else idx
        batch_arange = torch.arange(batch_size, device=device)

        # Augmented model input selected by ALINE.
        x_t = batch.query_x[batch_arange, idx_flat]  # [B, dim_x]

        # Physical simulator design.
        xi_t = _extract_xi_from_x(
            x_t,
            experiment=experiment,
            xi_position=xi_position,
        )  # [B, dim_xi]

        batch = experiment.update_batch(batch, idx)
        batch = _move_batch_to_device(batch, device)

        if hasattr(batch, "current_y"):
            y_t = batch.current_y
        else:
            y_t = batch.context_y[:, -1]

        xs.append(x_t)
        xis.append(xi_t)
        ys.append(y_t)

    theta_0 = batch.theta if hasattr(batch, "theta") else batch.target_all

    x = torch.stack(xs, dim=1)    # [B, T, dim_x]
    xi = torch.stack(xis, dim=1)  # [B, T, dim_xi]
    y = torch.stack(ys, dim=1)    # [B, T, dim_y]

    if return_augmented_x:
        return theta_0, x, xi, y

    return theta_0, xi, y


# ---------------------------------------------------------------------
# Sequential likelihood for SimplePendulum
# ---------------------------------------------------------------------

@torch.no_grad()
def history_logp_pendulum(experiment, theta, xi, y, stepwise=False):
    """
    Computes log p(y_{1:T} | theta, xi_{1:T}) for SimplePendulum.

    Args
    ----
    theta:
        [B, dim_theta] or [S, B, dim_theta]

    xi:
        [B, T, dim_xi]
        True physical controls.

    y:
        [B, T, dim_y]

    stepwise:
        if False: returns cumulative logp over full trajectory
        if True: returns cumulative logp at each time step

    Returns
    -------
    if theta is [B, D]:
        [B] if not stepwise else [B, T]

    if theta is [S, B, D]:
        [S, B] if not stepwise else [S, B, T]
    """

    device = xi.device
    B, T, dim_xi = xi.shape
    dim_y = y.shape[-1]

    expected_dim_xi = _get_dim_xi(experiment)
    if dim_xi != expected_dim_xi:
        raise ValueError(
            f"history_logp_pendulum expected xi.shape[-1]={expected_dim_xi}, "
            f"got {dim_xi}. You probably passed augmented x instead of xi."
        )

    theta = theta.to(device)
    xi = xi.to(device)
    y = y.to(device)

    if theta.ndim == 2:
        # theta: [B, dim_theta]
        y_prev = _initial_y(experiment, B, device)

        logps = []

        for t in range(T):
            logp_t = experiment.simulator._outcome_logp(
                y=y[:, t],
                theta=theta,
                design=xi[:, t],
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
            xi_t = xi[:, t].unsqueeze(0).expand(S, B, dim_xi)

            logp_t = experiment.simulator._outcome_logp(
                y=y_t,
                theta=theta,
                design=xi_t,
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
    xi,
    y,
    L=10000,
    batch_size=None,
    stepwise=False,
    L_chunk=2048,
):
    """
    PCE / NMC bounds for SimplePendulum.

    Args
    ----
    theta_0:
        [B, dim_theta]

    xi:
        [B, T, dim_xi]

    y:
        [B, T, dim_y]

    L:
        number of contrastive theta samples

    stepwise:
        False -> returns [B]
        True  -> returns [B, T]

    Returns
    -------
    pce:
        [B] or [B, T]

    nmc:
        [B] or [B, T]
    """

    device = xi.device
    theta_0 = theta_0.to(device)
    xi = xi.to(device)
    y = y.to(device)

    B = xi.shape[0]

    # True likelihood log p(y | theta_0, xi)
    logp_true = history_logp_pendulum(
        experiment=experiment,
        theta=theta_0,
        xi=xi,
        y=y,
        stepwise=stepwise,
    )  # [B] or [B, T]

    contrastive_lse = None
    n_done = 0

    while n_done < L:
        l_now = min(L_chunk, L - n_done)

        theta_c = _sample_theta(experiment, (l_now, B)).to(device)
        # theta_c: [l_now, B, dim_theta]

        logp_c = history_logp_pendulum(
            experiment=experiment,
            theta=theta_c,
            xi=xi,
            y=y,
            stepwise=stepwise,
        )  # [l_now, B] or [l_now, B, T]

        lse_c = torch.logsumexp(logp_c, dim=0)  # [B] or [B, T]

        if contrastive_lse is None:
            contrastive_lse = lse_c
        else:
            contrastive_lse = torch.logaddexp(contrastive_lse, lse_c)

        n_done += l_now

    pce_den_lse = torch.logaddexp(logp_true, contrastive_lse)

    log_L_plus_1 = torch.log(torch.tensor(float(L + 1), device=device))
    log_L = torch.log(torch.tensor(float(L), device=device))

    pce = log_L_plus_1 + logp_true - pce_den_lse
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
    xi_position="first",
):
    """
    Replacement for eval_boed, compatible with SimplePendulum
    when dim_x and dim_xi are dissociated.

    ALINE sees augmented x of dimension dim_x.
    The simulator likelihood uses xi of dimension dim_xi.
    """

    model.eval()

    pce_list = []
    nmc_list = []

    max_step = (M + batch_size - 1) // batch_size

    for step in range(max_step):
        b_now = min(batch_size, M - step * batch_size)

        theta_0, x_aug, xi, y = get_traces_pendulum(
            model=model,
            experiment=experiment,
            T=T,
            batch_size=b_now,
            time_token=time_token,
            xi_position=xi_position,
            return_augmented_x=True,
        )

        # Optional but useful sanity check on first batch.
        if step == 0:
            print("x_aug.shape:", tuple(x_aug.shape))
            print("xi.shape:", tuple(xi.shape))
            print("y.shape:", tuple(y.shape))
            print("theta_0.shape:", tuple(theta_0.shape))
            print("experiment.dim_x:", getattr(experiment, "dim_x", None))
            print("experiment.dim_xi:", getattr(experiment, "dim_xi", None))

        pce, nmc = compute_EIG_from_history_pendulum(
            experiment=experiment,
            theta_0=theta_0,
            xi=xi,
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
