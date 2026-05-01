import os
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import OmegaConf, DictConfig

from model import Aline


# ---------------------------------------------------------------------
# Downstream model
# ---------------------------------------------------------------------

class HistoryMLPThetaPredictor(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        T: int,
        n_sources: int = 2,
        source_dim: int = 2,
        hidden_dims=(512, 256, 128),
        activation="SiLU",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.T = T
        self.n_sources = n_sources
        self.source_dim = source_dim
        self.theta_dim = n_sources * source_dim

        input_dim = T * (dim_x + dim_y)

        act_cls = getattr(nn, activation)

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, self.theta_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, xi_history, y_history):
        h = torch.cat([xi_history, y_history], dim=-1)
        h = h.flatten(start_dim=1)
        theta_pred = self.net(h)
        theta_pred = theta_pred.view(
            -1, self.n_sources, self.source_dim
        )
        return theta_pred


# ---------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------

def canonicalize_sources_by_norm(theta):
    """
    theta: [B, K, D]
    """
    norms = torch.linalg.norm(theta, dim=-1)
    order = torch.argsort(norms, dim=1)
    gather_idx = order.unsqueeze(-1).expand_as(theta)
    return torch.gather(theta, dim=1, index=gather_idx)


def two_source_mse_per_sample(theta_pred, theta_true):
    """
    Permutation-invariant MSE for two sources.

    Returns:
        [B]
    """
    loss_id = ((theta_pred - theta_true) ** 2).mean(dim=(1, 2))

    theta_true_swapped = theta_true[:, [1, 0], :]
    loss_swap = ((theta_pred - theta_true_swapped) ** 2).mean(dim=(1, 2))

    return torch.minimum(loss_id, loss_swap)


def two_source_logmse_per_sample_old(theta_pred, theta_true, eps=1e-6):
    """
    Same canonicalized log-MSE as in your training code.

    Returns:
        [B]
    """
    theta_pred = canonicalize_sources_by_norm(theta_pred)
    theta_true = canonicalize_sources_by_norm(theta_true)

    err = theta_pred - theta_true
    loss = err ** 2 + torch.log(err ** 2 + eps)
    loss = loss.mean(dim=(1, 2))

    return loss


def two_source_logmse_per_sample(theta_pred, theta_true, eps=1e-6):
    """
    Permutation-invariant log-MSE-style metric for two 2D sources.

    Per-coordinate loss:
        err**2 + log(err**2 + eps)

    theta_pred: [B, 2, D]
    theta_true: [B, 2, D]

    Returns
    -------
    loss:
        [B]
    """

    err_id = theta_pred - theta_true
    loss_id = err_id ** 2 + torch.log(err_id ** 2 + eps)
    loss_id = loss_id.mean(dim=(1, 2))  # [B]

    theta_true_swapped = theta_true[:, [1, 0], :]
    err_swap = theta_pred - theta_true_swapped
    loss_swap = err_swap ** 2 + torch.log(err_swap ** 2 + eps)
    loss_swap = loss_swap.mean(dim=(1, 2))  # [B]

    return torch.minimum(loss_id, loss_swap)


def metric_mean_and_se(values):
    """
    values: [N]
    """
    mean = values.mean()
    se = values.std(unbiased=True) / values.numel() ** 0.5
    return mean.item(), se.item()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def load_aline_state_dict(aline_model, aline_path, device, strict=True):
    ckpt = torch.load(aline_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = aline_model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded ALINE checkpoint: {aline_path}")
    print(f"ALINE missing keys: {missing}")
    print(f"ALINE unexpected keys: {unexpected}")


def build_aline_and_experiment(cfg, device):
    experiment = hydra.utils.instantiate(cfg.task)

    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)

    aline_model = Aline(embedder, encoder, head).to(device)

    return aline_model, experiment


def build_downstream_from_cfg(cfg, device):
    downstream_net = HistoryMLPThetaPredictor(
        dim_x=cfg.task.dim_x,
        dim_y=cfg.task.dim_y,
        T=cfg.T,
        n_sources=cfg.task.K,
        source_dim=cfg.task.dim_x,
        hidden_dims=tuple(cfg.downstream.hidden_dims),
        activation=cfg.downstream.activation,
        dropout=cfg.downstream.dropout,
    ).to(device)

    return downstream_net


# ---------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------

@torch.inference_mode()
def rollout_aline_policy_fast(cfg, aline_model, experiment, batch_size, T, device):
    """
    Returns:
        xi_history: [B, T, dim_x]
        y_history:  [B, T, dim_y]
        theta:      [B, K, dim_x]
    """

    aline_model.eval()

    batch = experiment.sample_batch(batch_size)
    batch = move_batch_to_device(batch, device)

    B = batch.query_x.shape[0]
    dim_x = batch.query_x.shape[-1]
    dim_y = batch.context_y.shape[-1]

    xi_history = torch.empty(B, T, dim_x, device=device)
    y_history = torch.empty(B, T, dim_y, device=device)

    batch_indices = torch.arange(B, device=device)

    if getattr(cfg, "time_token", False):
        time_tokens = torch.arange(T, device=device, dtype=torch.float32) / T

    for t in range(T):
        if getattr(cfg, "time_token", False):
            batch.t = time_tokens[t:t + 1]

        pred = aline_model(batch)

        idx_update = pred.design_out.idx.long()

        if idx_update.ndim == 1:
            idx_update = idx_update.unsqueeze(1)
        elif idx_update.ndim > 2:
            idx_update = idx_update.view(idx_update.shape[0], -1)

        idx_flat = idx_update.squeeze(1)

        xi_history[:, t] = batch.query_x[batch_indices, idx_flat]

        batch = experiment.update_batch(batch, idx_update)
        batch = move_batch_to_device(batch, device)

        y_history[:, t] = batch.context_y[:, -1, :]

    if hasattr(batch, "target_all"):
        theta = batch.target_all
    elif hasattr(batch, "theta"):
        theta = batch.theta
    else:
        raise AttributeError("Batch has neither `target_all` nor `theta`.")

    theta = theta.to(device)
    theta = theta.view(B, cfg.task.K, cfg.task.dim_x)

    return xi_history, y_history, theta


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.inference_mode()
def evaluate_downstream(
    cfg,
    aline_model,
    downstream_net,
    experiment,
    batch_size,
    num_batches,
    T,
    device,
    eps_logmse=1e-6,
):

    aline_model.eval()
    downstream_net.eval()

    original_prior = getattr(experiment, "prior", None)

    mse_values = []
    logmse_values = []

    for _ in range(num_batches):
        xi_history, y_history, theta = rollout_aline_policy_fast(
            cfg=cfg,
            aline_model=aline_model,
            experiment=experiment,
            batch_size=batch_size,
            T=T,
            device=device,
        )

        theta_pred = downstream_net(xi_history, y_history)

        mse_batch = two_source_mse_per_sample(theta_pred, theta)
        logmse_batch = two_source_logmse_per_sample(
            theta_pred,
            theta,
            eps=eps_logmse,
        )

        mse_values.append(mse_batch.detach().cpu())
        logmse_values.append(logmse_batch.detach().cpu())

    mse_values = torch.cat(mse_values, dim=0)
    logmse_values = torch.cat(logmse_values, dim=0)

    mse_mean, mse_se = metric_mean_and_se(mse_values)
    logmse_mean, logmse_se = metric_mean_and_se(logmse_values)

    return {
        f"/mse": mse_mean,
        f"/mse_se": mse_se,
        f"/logmse": logmse_mean,
        f"/logmse_se": logmse_se,
        f"/n_examples": mse_values.numel(),
    }


def evaluate_downstream_checkpoint(
    downstream_ckpt_path,
    device,
    batch_size=1024,
    num_batches=20,
    T_eval=None,
    strict_aline=True,
    strict_downstream=True,
):
    downstream_ckpt_path = Path(downstream_ckpt_path)
    ckpt = torch.load(downstream_ckpt_path, map_location=device)

    if "config" not in ckpt:
        raise KeyError(f"No `config` found in {downstream_ckpt_path}")

    cfg = OmegaConf.create(ckpt["config"])

    if T_eval is None:
        T_eval = cfg.T

    aline_path = ckpt.get("aline_path", None)
    if aline_path is None:
        aline_path = getattr(cfg, "pretrained_aline_path", None)

    if aline_path is None:
        raise ValueError(
            "Could not infer ALINE checkpoint path. Expected `aline_path` in .pt "
            "or `cfg.pretrained_aline_path`."
        )

    aline_model, experiment = build_aline_and_experiment(cfg, device)

    load_aline_state_dict(
        aline_model=aline_model,
        aline_path=aline_path,
        device=device,
        strict=strict_aline,
    )

    downstream_net = build_downstream_from_cfg(cfg, device)

    missing, unexpected = downstream_net.load_state_dict(
        ckpt["downstream_net_state_dict"],
        strict=strict_downstream,
    )

    print(f"Loaded downstream checkpoint: {downstream_ckpt_path}")
    print(f"Downstream missing keys: {missing}")
    print(f"Downstream unexpected keys: {unexpected}")

    eps_logmse = getattr(cfg.downstream, "eps_loss", 1e-6)

    results = {
        "checkpoint": str(downstream_ckpt_path),
        "aline_path": str(aline_path),
        "T_eval": int(T_eval),
    }

    metrics = evaluate_downstream(
            cfg=cfg,
            aline_model=aline_model,
            downstream_net=downstream_net,
            experiment=experiment,
            batch_size=batch_size,
            num_batches=num_batches,
            T=T_eval,
            device=device,
            eps_logmse=eps_logmse,
        )

    results.update(metrics)

    return results


def find_downstream_checkpoints(path):
    path = Path(path)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(path.rglob("downstream_theta_predictor.pt"))

    raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to one downstream_theta_predictor.pt or to a directory containing many.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--T_eval", type=int, default=None)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print("Using GPU:", torch.cuda.get_device_name(device))

    ckpt_paths = find_downstream_checkpoints(args.path)

    print(f"Found {len(ckpt_paths)} downstream checkpoint(s).")

    all_results = []

    for ckpt_path in ckpt_paths:
        print("\n" + "=" * 80)
        print(f"Evaluating {ckpt_path}")
        print("=" * 80)

        results = evaluate_downstream_checkpoint(
            downstream_ckpt_path=ckpt_path,
            device=device,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            T_eval=args.T_eval,
        )

        all_results.append(results)

        print("\nResults:")
        for k, v in results.items():
            print(f"{k}: {v}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()