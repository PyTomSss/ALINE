import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os
import random
import numpy as np
import time

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from model import Aline
from utils import create_logger, set_seed, save_state_dict, compute_ll, load_checkpoint, save_checkpoint, create_target_mask, select_targets_by_mask, set_layerwise_lr, eval_boed


os.environ["HYDRA_FULL_ERROR"] = "1"

class HistoryMLPThetaPredictor(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        T,
        dim_theta,
        hidden_dims=(256, 256),
        activation="silu",
        dropout=0.0,
    ):
        super().__init__()

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.T = T
        self.dim_theta = dim_theta

        input_dim = T * (dim_x + dim_y)

        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, dim_theta))

        self.net = nn.Sequential(*layers)

    def forward(self, x_history, y_history):
        """
        x_history: [B, T, dim_x]
        y_history: [B, T, dim_y]

        returns:
            theta_pred: [B, dim_theta]
        """

        B, T, _ = x_history.shape

        assert T == self.T, f"Expected T={self.T}, got T={T}"

        h = torch.cat([x_history, y_history], dim=-1)
        h = h.reshape(B, T * (self.dim_x + self.dim_y))

        theta_pred = self.net(h)

        return theta_pred
    

# -------------- Utils ------------ #
@torch.inference_mode()
def rollout_aline_policy_fast(cfg, aline_model, experiment, batch_size, T, device):
    """
    Rollout frozen ALINE policy.

    Returns:
        x_history: [B, T, dim_x]
            Usually augmented ALINE input:
            x_t = [xi_t, y_{t-1}, time_t]

        y_history: [B, T, dim_y]
            Usually observed increments delta_y_t, or raw y_t depending on wrapper.

        theta: [B, dim_theta]
    """

    aline_model.eval()

    batch = experiment.sample_batch(batch_size)

    B = batch.query_x.shape[0]
    dim_x = batch.query_x.shape[-1]
    dim_y = batch.context_y.shape[-1]

    x_history = torch.empty(B, T, dim_x, device=device)
    y_history = torch.empty(B, T, dim_y, device=device)

    batch_indices = torch.arange(B, device=device)

    if cfg.time_token:
        time_tokens = torch.arange(T, device=device, dtype=torch.float32) / T

    for t in range(T):
        if cfg.time_token:
            batch.t = time_tokens[t:t + 1]

        pred = aline_model(batch)

        idx_update = pred.design_out.idx.long()

        if idx_update.ndim == 1:
            idx_update = idx_update.unsqueeze(1)
        elif idx_update.ndim > 2:
            idx_update = idx_update.view(idx_update.shape[0], -1)

        idx_flat = idx_update.squeeze(1)

        selected_x = batch.query_x[batch_indices, idx_flat]
        x_history[:, t] = selected_x

        batch = experiment.update_batch(batch, idx_update)

        y_history[:, t] = batch.context_y[:, -1, :]

    theta = batch.target_all if hasattr(batch, "target_all") else batch.theta

    return x_history, y_history, theta


def theta_mse_loss(theta_pred, theta_true):
    """
    theta_pred: [B, dim_theta]
    theta_true: [B, dim_theta]
    """
    return ((theta_pred - theta_true) ** 2).mean()


def theta_logmse_loss(theta_pred, theta_true, eps=1e-6):
    """
    Log-MSE-style loss for ordered theta vectors.

    Per-coordinate loss:
        err**2 + log(err**2 + eps)

    theta_pred: [B, dim_theta]
    theta_true: [B, dim_theta]
    """
    err2 = (theta_pred - theta_true) ** 2
    loss = err2 + torch.log(err2 + eps)
    return loss.mean()


def theta_weighted_mse_loss(theta_pred, theta_true, weights):
    """
    Weighted MSE over theta coordinates.

    theta_pred: [B, dim_theta]
    theta_true: [B, dim_theta]
    weights:    [dim_theta] or list/tuple of length dim_theta

    Example:
        weights = torch.tensor([1.0, 10.0, 1.0], device=device)
    """
    if not torch.is_tensor(weights):
        weights = torch.tensor(weights, device=theta_pred.device, dtype=theta_pred.dtype)
    else:
        weights = weights.to(device=theta_pred.device, dtype=theta_pred.dtype)

    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")

    if weights.shape[0] != theta_pred.shape[-1]:
        raise ValueError(
            f"weights length must match dim_theta={theta_pred.shape[-1]}, "
            f"got {weights.shape[0]}"
        )

    err2 = (theta_pred - theta_true) ** 2

    # [B, dim_theta] * [dim_theta]
    loss = err2 * weights

    return loss.mean()


# -------------- Training ------------ #
def train_downstream_from_pretrained_aline(
    cfg,
    logger,
    aline_model,
    downstream_net,
    experiment,
    batch_size,
    T,
    max_epoch,
    device,
):
    """
    Train only the downstream theta predictor.
    ALINE is frozen and used as acquisition/design policy.
    """

    aline_model.eval()
    for p in aline_model.parameters():
        p.requires_grad_(False)

    optimizer = optim.Adam(
        downstream_net.parameters(),
        lr=cfg.downstream.lr,
        weight_decay=cfg.downstream.weight_decay,
    )

    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.downstream.gamma,
    )

    losses = []

    for epoch in range(max_epoch):
        downstream_net.train()
        optimizer.zero_grad()

        x_history, y_history, theta = rollout_aline_policy_fast(
            cfg=cfg,
            aline_model=aline_model,
            experiment=experiment,
            batch_size=batch_size,
            T=T,
            device=device,
        )

        theta = theta.to(device)
        theta = theta.view(batch_size, cfg.task.n_target_theta)

        theta_pred = downstream_net(x_history, y_history)

        if cfg.downstream.loss == "mse":
            loss = theta_mse_loss(theta_pred, theta)

        elif cfg.downstream.loss == "logmse":
            loss = theta_logmse_loss(
                theta_pred,
                theta,
                eps=getattr(cfg.downstream, "eps_loss", 1e-6),
            )

        elif cfg.downstream.loss == "weighted_mse":
            loss = theta_weighted_mse_loss(
                theta_pred,
                theta,
                weights=cfg.downstream.theta_weights,
            )

        else:
            raise ValueError(f"Unknown downstream loss: {cfg.downstream.loss}")
        
        loss.backward()

        if getattr(cfg.downstream, "clip_grads", False):
            clip_grad_norm_(downstream_net.parameters(), max_norm=1.0)

        optimizer.step()

        if (epoch + 1) % cfg.downstream.annealing_freq == 0:
            scheduler.step()

        losses.append(loss.item())

        if cfg.wandb.use_wandb and (epoch + 1) % 50 == 0:
            wandb.log(
                {
                    "downstream/loss": loss.item(),
                    "downstream/lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if (epoch + 1) % cfg.downstream.verbose == 0:
            logger.info(
                f"[Downstream] Epoch {epoch + 1}/{max_epoch} | "
                f"loss={loss.item():.6f}"
            )

    return losses


# -------------- Main ------------ #
@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    # Device setup
    requested_device = str(cfg.device)

    print("Requested device:", requested_device)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.version.cuda:", torch.version.cuda)

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"cfg.device={requested_device}, but CUDA is not available. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                f"torch.version.cuda={torch.version.cuda}"
            )

        if requested_device == "cuda":
            requested_device = "cuda:0"

        device = torch.device(requested_device)
        torch.cuda.set_device(device.index)
    else:
        device = torch.device("cpu")

    cfg.device = str(device)
    torch.set_default_device(device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        print("Using GPU:", torch.cuda.get_device_name(device))

    if cfg.fix_seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    if cfg.min_T > cfg.T:
        cfg.min_T = cfg.T
    
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name=cfg.task.name)
    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))


    # -------------- WandB --------------- #
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
        )
        
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    experiment = hydra.utils.instantiate(cfg.task)
    logger.info(f"Task: {experiment}")

    if hasattr(experiment, 'meta_dataset'):
        logger.info(f"Using HPO-B meta-dataset: {experiment.meta_dataset}")
        logger.info(f"Input dimension: {experiment.dim_x}")
        logger.info(f"Number of datasets: {experiment.hpob.n_dataset}")

        # Update config with actual dimensions from the dataset
        if cfg.task.dim_x != experiment.dim_x:
            logger.info(
                f"Updating dim_x from config value {cfg.task.dim_x} to actual dataset dimension {experiment.dim_x}")
            cfg.task.dim_x = experiment.dim_x

        if cfg.task.dim_y != experiment.dim_y:
            logger.info(
                f"Updating dim_y from config value {cfg.task.dim_y} to actual dataset dimension {experiment.dim_y}")
            cfg.task.dim_y = experiment.dim_y



    # -------------- Model --------------- #
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = Aline(embedder, encoder, head)
    device = torch.device(cfg.device)
    model = model.to(device)

    if getattr(cfg, "pretrained_aline_path", None) is not None:
        logger.info(f"Loading pretrained ALINE from {cfg.pretrained_aline_path}")

        ckpt = torch.load(cfg.pretrained_aline_path, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=True)

        logger.info("Pretrained ALINE loaded successfully.")
    else:
        raise ValueError("cfg.pretrained_aline_path must be provided for downstream training.")
    
    
    logger.info(model)
    logger.info(f"Model device: {next(model.parameters()).device}")

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)


    # -------------- Downstream net --------------- #
    downstream_net = HistoryMLPThetaPredictor(
        dim_x=cfg.task.dim_x,
        dim_y=cfg.task.dim_y,
        T=cfg.T,
        dim_theta=cfg.task.n_target_theta,
        hidden_dims=tuple(cfg.downstream.hidden_dims),
        activation=cfg.downstream.activation,
        dropout=cfg.downstream.dropout,
    ).to(device)

    logger.info(downstream_net)
    logger.info(f"Downstream model device: {next(downstream_net.parameters()).device}")


    # -------------- Training Procedure --------------- #
    train_downstream_from_pretrained_aline(
        cfg=cfg,
        logger=logger,
        aline_model=model,
        downstream_net=downstream_net,
        experiment=experiment,
        batch_size=cfg.batch_size,
        T=cfg.T,
        max_epoch=cfg.downstream.max_epoch,
        device=device,
    )

    downstream_save_path = os.path.join(cfg.output_dir, "downstream_theta_predictor.pt")

    torch.save(
        {
            "downstream_net_state_dict": downstream_net.state_dict(),
            "aline_path": cfg.pretrained_aline_path,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        downstream_save_path,
    )

    logger.info(f"Downstream model saved at {downstream_save_path}")
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")


    # -------------- EIG Evaluation --------------- #
    if cfg.eval.EIG:
    
        experiment.n_query_init = cfg.eval.n_query_final

        bounds = eval_boed(model, experiment, cfg.eval.T_final - cfg.task.n_context_init, cfg.eval.L_final, cfg.eval.M_final, cfg.eval.batch_size_final, cfg.time_token, stepwise=True)

        logger.info(bounds)
        logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_err'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_err'][cfg.T-1]:.3f}")

       
        save_path = os.path.join(cfg.output_dir, "eval", f"{cfg.file_name.split('.')[0]}_N{cfg.eval.n_query_final}_T{cfg.eval.T_final}.tar")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(bounds, save_path)
        logger.info(f"Bounds have been saved at {save_path}.")
                           


if __name__ == '__main__':
    main()
