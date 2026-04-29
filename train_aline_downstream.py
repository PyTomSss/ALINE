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
    """
    MLP downstream that predicts theta from flattened design/observation history.

    Input:
        xi_history: [B, T, dim_x]
        y_history:  [B, T, dim_y]

    Output:
        theta_pred: [B, n_sources, source_dim]
    """

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
        """
        xi_history: [B, T, dim_x]
        y_history:  [B, T, dim_y]
        """

        h = torch.cat([xi_history, y_history], dim=-1)  # [B, T, dim_x + dim_y]
        h = h.flatten(start_dim=1)                      # [B, T * (dim_x + dim_y)]

        theta_pred = self.net(h)                        # [B, 4]
        theta_pred = theta_pred.view(
            -1, self.n_sources, self.source_dim
        )                                               # [B, 2, 2]

        return theta_pred    


@torch.no_grad()
def rollout_aline_policy(cfg, aline_model, experiment, batch_size, T, device):
    """
    Rolls out a frozen ALINE policy and returns histories.

    Returns:
        xi_history: [B, T, dim_x]
        y_history:  [B, T, dim_y]
        theta:      [B, n_sources, source_dim] or [B, theta_dim]
    """

    aline_model.eval()

    batch = experiment.sample_batch(batch_size)

    xi_hist = []
    y_hist = []

    for t in range(T):
        if cfg.time_token:
            batch.t = torch.tensor([t / T], device=device)

        pred = aline_model.forward(batch)
        design_out = pred.design_out

        # Store selected design before updating batch.
        idx = design_out.idx.squeeze(-1)  # [B]

        batch_indices = torch.arange(batch_size, device=device)
        xi_t = batch.query_x[batch_indices, idx]  # [B, dim_x]

        batch = experiment.update_batch(batch, design_out.idx)

        # In your HiddenLocationFinding-style task, update_batch should append
        # the newly observed y to context_y. The last context_y is y_t.
        y_t = batch.context_y[:, -1, :]  # [B, dim_y]

        xi_hist.append(xi_t)
        y_hist.append(y_t)

    xi_history = torch.stack(xi_hist, dim=1)  # [B, T, dim_x]
    y_history = torch.stack(y_hist, dim=1)    # [B, T, dim_y]

    theta = batch.theta if hasattr(batch, "theta") else batch.target_all

    return xi_history, y_history, theta


def two_source_mse_loss(theta_pred, theta_true):
    """
    Permutation-invariant MSE for two 2D sources.

    theta_pred: [B, 2, 2]
    theta_true: [B, 2, 2]
    """

    loss_id = ((theta_pred - theta_true) ** 2).mean(dim=(1, 2))

    theta_true_swapped = theta_true[:, [1, 0], :]
    loss_swap = ((theta_pred - theta_true_swapped) ** 2).mean(dim=(1, 2))

    loss = torch.minimum(loss_id, loss_swap)

    return loss.mean()

def two_source_logmse_loss(theta_pred, theta_true, eps=1e-6):
    """
    Permutation-invariant log-MSE-style loss for two 2D sources.

    Per-coordinate loss:
        err**2 + log(err**2 + eps)

    theta_pred: [B, 2, 2]
    theta_true: [B, 2, 2]
    """

    err_id = theta_pred - theta_true
    loss_id = (err_id ** 2 + torch.log(err_id ** 2 + eps)).mean(dim=(1, 2))

    theta_true_swapped = theta_true[:, [1, 0], :]
    err_swap = theta_pred - theta_true_swapped
    loss_swap = (err_swap ** 2 + torch.log(err_swap ** 2 + eps)).mean(dim=(1, 2))

    loss = torch.minimum(loss_id, loss_swap)

    return loss.mean()


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
    Train only the downstream MLP.
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

        xi_history, y_history, theta = rollout_aline_policy(
            cfg=cfg,
            aline_model=aline_model,
            experiment=experiment,
            batch_size=batch_size,
            T=T,
            device=device,
        )

        theta = theta.to(device)

        # Expected target shape: [B, 2, 2].
        theta = theta.view(batch_size, cfg.task.K, cfg.task.dim_x)

        theta_pred = downstream_net(xi_history, y_history)

        if cfg.downstream.loss == "mse":
            loss = two_source_mse_loss(theta_pred, theta)
        elif cfg.downstream.loss == "logmse":
            loss = two_source_logmse_loss(
                theta_pred,
                theta,
                eps=getattr(cfg.downstream, "eps_loss", 1e-6),
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

        if (epoch + 1) % cfg.verbose == 0:
            logger.info(
                f"[Downstream] Epoch {epoch + 1}/{max_epoch} | "
                f"loss={loss.item():.6f}"
            )

    return losses


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

        # If cfg.device == "cuda", make it explicit.
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

    # Setting random seed
    if cfg.fix_seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    # Ensure min_T is not larger than T
    if cfg.min_T > cfg.T:
        cfg.min_T = cfg.T
    
    # Create logger
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
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # Data
    experiment = hydra.utils.instantiate(cfg.task)
    logger.info(f"Task: {experiment}")

    # For HPO tasks, validate that we have an HPO task and update config dimensions
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

    # Load pretrained ALINE weights
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
        n_sources=cfg.task.K,
        source_dim=cfg.task.dim_x,
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
    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")


    # -------------- EIG Evaluation --------------- #
    if cfg.eval.EIG:
        # Set a larger query size
        experiment.n_query_init = cfg.eval.n_query_final

        bounds = eval_boed(model, experiment, cfg.eval.T_final - cfg.task.n_context_init, cfg.eval.L_final, cfg.eval.M_final, cfg.eval.batch_size_final, cfg.time_token, stepwise=True)

        logger.info(bounds)
        logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_err'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_err'][cfg.T-1]:.3f}")

        # save bounds to file
        save_path = os.path.join(cfg.output_dir, "eval", f"{cfg.file_name.split('.')[0]}_N{cfg.eval.n_query_final}_T{cfg.eval.T_final}.tar")
        # make dir if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(bounds, save_path)
        logger.info(f"Bounds have been saved at {save_path}.")
                           


if __name__ == '__main__':
    main()
