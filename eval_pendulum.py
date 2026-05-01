import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from model import Aline
from utils import eval_boed 
from utils.eval import eval_boed_pendulum


# ---------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------

def load_hydra_cfg(run_dir: str):
    """
    run_dir should be the Hydra output directory containing .hydra/config.yaml.
    Example:
        outputs/2026-04-28/12-34-56
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find Hydra config at {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.output_dir = str(run_dir)
    return cfg


def setup_device(cfg, device: Optional[str] = None):
    if device is None:
        device = str(cfg.device)

    if device == "cuda":
        device = "cuda:0"

    device = torch.device(device)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        torch.cuda.set_device(device)

    cfg.device = str(device)
    torch.set_default_device(device)

    return device


def build_experiment_and_model(cfg, device):
    """
    Rebuilds the ALINE experiment and model from the saved Hydra config.
    """
    experiment = hydra.utils.instantiate(cfg.task)

    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)

    model = Aline(embedder, encoder, head).to(device)
    model.eval()

    return experiment, model


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Robustly extracts model weights from either:
      - plain state_dict saved by save_state_dict
      - full checkpoint saved by save_checkpoint
    """
    if not isinstance(obj, dict):
        raise TypeError("Loaded object is not a dict-like checkpoint/state_dict.")

    # Case 1: plain model.state_dict()
    if all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # Case 2: common checkpoint keys
    possible_keys = [
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "weights",
    ]

    for key in possible_keys:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    raise KeyError(
        f"Could not find model state dict. Available keys: {list(obj.keys())}"
    )


def load_aline_weights(model, weight_path: str, device, strict: bool = True):
    weight_path = Path(weight_path)

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    return model


def get_dim_xi(cfg, experiment):
    if hasattr(cfg.task, "dim_xi"):
        return int(cfg.task.dim_xi)
    if hasattr(experiment, "dim_xi"):
        return int(experiment.dim_xi)
    # Backward compatibility: old setup dim_x == dim_xi
    return int(cfg.task.dim_x)


def extract_xi_from_model_x(x, cfg, experiment, xi_position: str = "first"):
    """
    Extract the true simulator design xi from the augmented ALINE x.

    x:
        [..., dim_x]

    Returns:
        xi:
            [..., dim_xi]

    Convention assumed here:
        x = concat([xi, auxiliary_context_features])

    If your convention is instead x = concat([auxiliary_context_features, xi]),
    set xi_position="last".
    """

    dim_xi = get_dim_xi(cfg, experiment)

    if x.shape[-1] == dim_xi:
        return x

    if x.shape[-1] < dim_xi:
        raise ValueError(
            f"Cannot extract dim_xi={dim_xi} from x.shape[-1]={x.shape[-1]}"
        )

    if xi_position == "first":
        return x[..., :dim_xi]

    elif xi_position == "last":
        return x[..., -dim_xi:]

    else:
        raise ValueError(f"Unknown xi_position: {xi_position}")

# ---------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------

@torch.no_grad()
def rollout_aline_pendulum(
    model,
    experiment,
    cfg,
    T: int,
    batch_size: int = 1,
    n_query: Optional[int] = None,
    time_token: bool = False,
    device: Optional[torch.device] = None,
    theta: Optional[torch.Tensor] = None,
    xi_position: str = "first",
):
    """
    Runs one ALINE active experiment rollout.

    Important distinction:
        model_designs:
            selected augmented x seen by ALINE, shape [B, T, dim_x]

        simulator_designs:
            true simulator controls xi, shape [B, T, dim_xi]

    Returns:
        dict with theta, model_designs, simulator_designs, observations,
        selected indices, log_probs.
    """

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    if n_query is not None:
        experiment.n_query_init = n_query

    batch = experiment.sample_batch(batch_size)

    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    if theta is not None:
        theta = theta.to(device)
        if theta.ndim == 1:
            theta = theta[None, :]
        assert theta.shape[0] == batch_size
        batch.theta = theta
        batch.target_all = theta

    model_xs = []
    xis = []
    ys = []
    idxs = []
    log_probs = []

    if hasattr(batch, "current_y"):
        y0 = batch.current_y.detach().cpu()
    else:
        y0 = batch.context_y[:, -1].detach().cpu()

    for t in range(T):
        if time_token:
            batch.t = torch.tensor([t / T], device=device)

        pred = model.forward(batch)

        design_out = pred.design_out
        idx = design_out.idx

        idx_squeezed = idx.squeeze(-1) if idx.ndim > 1 else idx
        batch_arange = torch.arange(batch_size, device=device)

        # This is the augmented ALINE input x.
        selected_model_x = batch.query_x[batch_arange, idx_squeezed]

        # This is the true physical design xi.
        selected_xi = extract_xi_from_model_x(
            selected_model_x,
            cfg=cfg,
            experiment=experiment,
            xi_position=xi_position,
        )

        batch = experiment.update_batch(batch, idx)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        model_xs.append(selected_model_x.detach().cpu())
        xis.append(selected_xi.detach().cpu())

        if hasattr(batch, "current_y"):
            ys.append(batch.current_y.detach().cpu())
        else:
            ys.append(batch.context_y[:, -1].detach().cpu())

        idxs.append(idx.detach().cpu())

        if hasattr(design_out, "log_prob"):
            log_probs.append(design_out.log_prob.detach().cpu())

    model_xs = torch.stack(model_xs, dim=1)   # [B, T, dim_x]
    xis = torch.stack(xis, dim=1)             # [B, T, dim_xi]
    ys = torch.stack(ys, dim=1)               # [B, T, dim_y]
    idxs = torch.stack(idxs, dim=1)

    out = {
        "theta": batch.theta.detach().cpu() if hasattr(batch, "theta") else batch.target_all.detach().cpu(),
        "x0": y0,
        "model_designs": model_xs,
        "simulator_designs": xis,
        "observations": ys,
        "idxs": idxs,
        "final_batch": batch,
    }

    if len(log_probs) > 0:
        out["log_probs"] = torch.stack(log_probs, dim=1)

    return out


@torch.no_grad()
def rollout_aline_double_pendulum(
    model,
    experiment,
    cfg,
    T: int,
    batch_size: int = 1,
    n_query: Optional[int] = None,
    time_token: bool = False,
    device: Optional[torch.device] = None,
    theta: Optional[torch.Tensor] = None,
    xi_position: str = "first",
):
    """
    Runs one ALINE active rollout for HiddenDoublePendulum.

    Returns:
        model_designs:     [B, T, dim_x]      augmented ALINE inputs
        simulator_designs: [B, T, dim_xi]     physical controls
        observations:      [B, T, dim_y]      physical states
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    if n_query is not None:
        experiment.n_query_init = n_query

    batch = experiment.sample_batch(batch_size)

    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    if theta is not None:
        theta = theta.to(device)
        if theta.ndim == 1:
            theta = theta[None, :]
        assert theta.shape[0] == batch_size

        batch.theta = theta
        batch.target_theta = theta

        if hasattr(experiment, "make_target"):
            batch.target_all = experiment.make_target(theta)
        else:
            batch.target_all = theta

    model_xs = []
    xis = []
    ys = []
    idxs = []
    log_probs = []

    y0 = batch.current_y.detach().cpu()

    for t in range(T):
        # Only use this if your ALINE architecture explicitly expects batch.t.
        # If the wrapper already includes time in query_x, you usually want this False.
        if time_token:
            batch.t = torch.tensor([t / T], device=device)

        pred = model.forward(batch)

        design_out = pred.design_out
        idx = design_out.idx

        idx_squeezed = idx.squeeze(-1) if idx.ndim > 1 else idx
        batch_arange = torch.arange(batch_size, device=device)

        selected_model_x = batch.query_x[batch_arange, idx_squeezed]

        selected_xi = extract_xi_from_model_x(
            selected_model_x,
            cfg=cfg,
            experiment=experiment,
            xi_position=xi_position,
        )

        batch = experiment.update_batch(batch, idx)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        model_xs.append(selected_model_x.detach().cpu())
        xis.append(selected_xi.detach().cpu())
        ys.append(batch.current_y.detach().cpu())
        idxs.append(idx.detach().cpu())

        if hasattr(design_out, "log_prob"):
            log_probs.append(design_out.log_prob.detach().cpu())

    model_xs = torch.stack(model_xs, dim=1)
    xis = torch.stack(xis, dim=1)
    ys = torch.stack(ys, dim=1)
    idxs = torch.stack(idxs, dim=1)

    out = {
        "theta": batch.theta.detach().cpu(),
        "x0": y0,
        "model_designs": model_xs,
        "simulator_designs": xis,
        "observations": ys,
        "idxs": idxs,
        "final_batch": batch,
    }

    if len(log_probs) > 0:
        out["log_probs"] = torch.stack(log_probs, dim=1)

    return out

# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_pendulum_rollout(
    rollout: Dict[str, torch.Tensor],
    batch_index: int = 0,
    title: str = "ALINE rollout on SimplePendulum",
    save_path: Optional[str] = None,
):
    designs = rollout["simulator_designs"][batch_index]  # [T, dim_xi]
    ys = rollout["observations"][batch_index]
    theta = rollout["theta"][batch_index]

    T = ys.shape[0]
    t_grid = torch.arange(1, T + 1)

    angle = ys[:, 0]
    velocity = ys[:, 1]
    control = designs[:, 0]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t_grid, angle, marker="o")
    axes[0].set_ylabel("angle")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_grid, velocity, marker="o")
    axes[1].set_ylabel("angular velocity")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(t_grid, control, where="mid")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("selected control xi")
    axes[2].grid(True, alpha=0.3)

    theta_str = ", ".join([f"{v:.3f}" for v in theta.tolist()])
    fig.suptitle(f"{title}\nθ = [{theta_str}]")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_double_pendulum_rollout(
    rollout: Dict[str, torch.Tensor],
    batch_index: int = 0,
    title: str = "ALINE rollout on DoublePendulum",
    save_path: Optional[str] = None,
):
    designs = rollout["simulator_designs"][batch_index]  # [T, 2]
    ys = rollout["observations"][batch_index]            # [T, 4]
    theta = rollout["theta"][batch_index]                # [4]

    T = ys.shape[0]
    t_grid = torch.arange(1, T + 1)

    q1 = ys[:, 0]
    q2 = ys[:, 1]
    q1d = ys[:, 2]
    q2d = ys[:, 3]

    xi1 = designs[:, 0]
    xi2 = designs[:, 1]

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(t_grid, q1, marker="o")
    axes[0].set_ylabel("q1")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_grid, q2, marker="o")
    axes[1].set_ylabel("q2")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_grid, q1d, marker="o")
    axes[2].set_ylabel("q1 dot")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_grid, q2d, marker="o")
    axes[3].set_ylabel("q2 dot")
    axes[3].grid(True, alpha=0.3)

    axes[4].step(t_grid, xi1, where="mid")
    axes[4].set_ylabel("xi1")
    axes[4].grid(True, alpha=0.3)

    axes[5].step(t_grid, xi2, where="mid")
    axes[5].set_xlabel("step")
    axes[5].set_ylabel("xi2")
    axes[5].grid(True, alpha=0.3)

    theta_str = ", ".join([f"{v:.3f}" for v in theta.tolist()])
    fig.suptitle(f"{title}\nθ = [{theta_str}]")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


def plot_phase_portrait(
    rollout: Dict[str, torch.Tensor],
    batch_index: int = 0,
    title: str = "Pendulum phase trajectory",
    save_path: Optional[str] = None,
):
    ys = rollout["observations"][batch_index]
    theta = rollout["theta"][batch_index]

    angle = ys[:, 0]
    velocity = ys[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(angle, velocity, marker="o")
    ax.set_xlabel("angle")
    ax.set_ylabel("angular velocity")
    ax.grid(True, alpha=0.3)

    theta_str = ", ".join([f"{v:.3f}" for v in theta.tolist()])
    ax.set_title(f"{title}\nθ = [{theta_str}]")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, ax


def plot_double_pendulum_phase_portraits(
    rollout: Dict[str, torch.Tensor],
    batch_index: int = 0,
    title: str = "Double pendulum phase trajectories",
    save_path: Optional[str] = None,
):
    ys = rollout["observations"][batch_index]
    theta = rollout["theta"][batch_index]

    q1 = ys[:, 0]
    q2 = ys[:, 1]
    q1d = ys[:, 2]
    q2d = ys[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(q1, q1d, marker="o")
    axes[0].set_xlabel("q1")
    axes[0].set_ylabel("q1 dot")
    axes[0].set_title("First joint")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(q2, q2d, marker="o")
    axes[1].set_xlabel("q2")
    axes[1].set_ylabel("q2 dot")
    axes[1].set_title("Second joint")
    axes[1].grid(True, alpha=0.3)

    theta_str = ", ".join([f"{v:.3f}" for v in theta.tolist()])
    fig.suptitle(f"{title}\nθ = [{theta_str}]")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes


# ---------------------------------------------------------------------
# Optional EIG evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_eig(
    model,
    experiment,
    cfg,
    T_eval=None,
    n_query=None,
    L=None,
    M=None,
    batch_size=None,
    stepwise=True,
    L_chunk=2048,
    xi_position: str = "first",
):
    if T_eval is None:
        T_eval = cfg.eval.T_final - cfg.task.n_context_init

    if n_query is not None:
        experiment.n_query_init = n_query
    elif hasattr(cfg.eval, "n_query_final"):
        experiment.n_query_init = cfg.eval.n_query_final

    if L is None:
        L = cfg.eval.L_final if hasattr(cfg.eval, "L_final") else cfg.eval.L

    if M is None:
        M = cfg.eval.M_final if hasattr(cfg.eval, "M_final") else cfg.eval.M

    if batch_size is None:
        batch_size = (
            cfg.eval.batch_size_final
            if hasattr(cfg.eval, "batch_size_final")
            else cfg.eval.batch_size
        )

    T_eval = 50
    L = 50000
    M = 500
    stepwise=False

    return eval_boed_pendulum(
        model=model,
        experiment=experiment,
        #cfg=cfg,
        T=T_eval,
        L=L,
        M=M,
        batch_size=batch_size,
        time_token=cfg.time_token,
        stepwise=stepwise,
        L_chunk=L_chunk,
        xi_position=xi_position,
    )


# ---------------------------------------------------------------------
# Example main
# ---------------------------------------------------------------------

def main():
    #run_dir = "checkpoints/61643_1/"
    run_dir = "checkpoints/76919_1/"
    #weight_path = "checkpoints/61643_1/checkpoints_after_burning/model/aae_Pendulum_d1_1_epoch_50000.pth"
    weight_path = "checkpoints/76919_1/model/aae_Pendulum_dx4_dxi1_T50_1.pth"


    cfg = load_hydra_cfg(run_dir)
    #device = setup_device(cfg, device="cuda:0")
    device = setup_device(cfg, device="cpu")

    experiment, model = build_experiment_and_model(cfg, device)
    model = load_aline_weights(model, weight_path, device=device, strict=True)

    if cfg.task.name == "DoublePendulum": 
        rollout = rollout_aline_double_pendulum(
            model=model,
            experiment=experiment,
            cfg=cfg,
            T=cfg.T - cfg.task.n_context_init,
            batch_size=1,
            n_query=cfg.task.n_query_init,
            time_token=False,
            device=device,
            xi_position="first",
        )

        plot_double_pendulum_rollout(
            rollout,
            save_path=Path(run_dir) / "eval" / "double_pendulum_rollout_timeseries.png",
        )

        plot_double_pendulum_phase_portraits(
            rollout,
            save_path=Path(run_dir) / "eval" / "double_pendulum_phase_portraits.png",
        )

        plt.show()

    else: 
    
        # Rollout visualization
        rollout = rollout_aline_pendulum(
            model=model,
            experiment=experiment,
            cfg=cfg,
            T=cfg.T - cfg.task.n_context_init,
            batch_size=1,
            n_query=cfg.task.n_query_init,
            time_token=cfg.time_token,
            device=device,
            xi_position="first",
        )

        plot_pendulum_rollout(
            rollout,
            save_path=Path(run_dir) / "eval" / "rollout_timeseries.png",
        )

        plot_phase_portrait(
            rollout,
            save_path=Path(run_dir) / "eval" / "rollout_phase.png",
        )

        plt.show()

    # Optional: EIG bounds
    bounds = evaluate_eig(model, experiment, cfg)
    print(bounds)


if __name__ == "__main__":
    main()