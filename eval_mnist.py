# eval_aline_masked_mnist_classification.py

import os
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf

from model import Aline
from utils import compute_ll

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

os.environ["HYDRA_FULL_ERROR"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("\n=== DEBUG IMPORT CONTEXT ===")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("CWD:", os.getcwd())
print("sys.path[:10]:")
for p in sys.path[:10]:
    print("  ", p)

try:
    import tasks
    print("tasks module:", tasks)
    print("tasks file:", getattr(tasks, "__file__", None))
except Exception:
    print("FAILED importing tasks")
    traceback.print_exc()

try:
    import tasks.masked_mnist as mm
    print("masked_mnist file:", mm.__file__)
    print("Has MaskedMNISTClassification:", hasattr(mm, "MaskedMNISTClassification"))
    print("MaskedMNISTClassification:", getattr(mm, "MaskedMNISTClassification", None))
except Exception:
    print("FAILED importing tasks.masked_mnist")
    traceback.print_exc()

print("=== END DEBUG IMPORT CONTEXT ===\n")


def _move_batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


def _repeat_first_dim(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=0)


def class_logits_from_posterior(
    posterior_out,
    num_classes: int = 10,
    use_mean_over_targets: bool = True,
):
    """
    Convert ALINE Gaussian-mixture posterior over one-hot labels into class logits.

    For each class c, define:

        logit_c = log p(one_hot(c) | history)

    using the same compute_ll used during training.

    posterior_out must contain:
        mixture_means
        mixture_stds
        mixture_weights

    Returns
    -------
    logits:
        [B, num_classes]
    """
    means = posterior_out.mixture_means
    stds = posterior_out.mixture_stds
    weights = posterior_out.mixture_weights

    device = means.device
    B = means.shape[0]

    eye = torch.eye(num_classes, device=device, dtype=means.dtype)  # [C, C]
    targets = eye.unsqueeze(0).expand(B, num_classes, num_classes)  # [B, C, C]
    targets_flat = targets.reshape(B * num_classes, num_classes)   # [B*C, C]

    means_rep = _repeat_first_dim(means, num_classes)
    stds_rep = _repeat_first_dim(stds, num_classes)
    weights_rep = _repeat_first_dim(weights, num_classes)

    # compute_ll returns per-target-dimension log-likelihood: [B*C, num_classes]
    target_ll = compute_ll(
        targets_flat,
        means_rep,
        stds_rep,
        weights_rep,
    )

    if use_mean_over_targets:
        class_scores = target_ll.mean(dim=-1)  # [B*C]
    else:
        class_scores = target_ll.sum(dim=-1)   # [B*C]

    logits = class_scores.view(B, num_classes)  # [B, C]
    return logits


@torch.no_grad()
def rollout_and_get_logits(
    cfg,
    model,
    experiment,
    batch_size: int,
    T: int,
    device: torch.device,
    num_classes: int = 10,
):
    """
    Sample a batch, run ALINE for T patch observations, then return final class logits.

    Returns
    -------
    logits:
        [B, num_classes]

    labels:
        [B]
    """
    batch = experiment.sample_batch(batch_size)
    batch = _move_batch_to_device(batch, device)

    for t in range(T):
        if cfg.time_token:
            batch.t = torch.tensor([t / T], device=device)

        pred = model.forward(batch)
        design_out = pred.design_out

        batch = experiment.update_batch(batch, design_out.idx)
        batch = _move_batch_to_device(batch, device)

    # Final posterior after T observed patches.
    if cfg.time_token:
        batch.t = torch.tensor([1.0], device=device)

    pred = model.forward(batch)
    posterior_out = pred.posterior_out

    logits = class_logits_from_posterior(
        posterior_out,
        num_classes=num_classes,
        use_mean_over_targets=True,
    )

    labels = batch.labels.long().to(device)

    return logits, labels


@torch.no_grad()
def evaluate_split(
    cfg,
    model,
    experiment,
    split: str,
    batch_size: int,
    num_batches: int,
    T: int,
    device: torch.device,
    num_classes: int = 10,
    deterministic_observations: bool = False,
):
    """
    Evaluate CE loss and accuracy on either train or test split.

    split:
        "train" or "test"

    deterministic_observations:
        If True, temporarily sets experiment.noise_scale = 0.0.
    """
    assert split in {"train", "test"}

    model.eval()

    original_prior = experiment.prior
    original_noise_scale = getattr(experiment, "noise_scale", None)

    if split == "test":
        if not hasattr(experiment, "prior_test"):
            raise AttributeError("experiment has no prior_test attribute.")
        experiment.prior = experiment.prior_test

    if deterministic_observations and original_noise_scale is not None:
        experiment.noise_scale = 0.0

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for _ in range(num_batches):
        logits, labels = rollout_and_get_logits(
            cfg=cfg,
            model=model,
            experiment=experiment,
            batch_size=batch_size,
            T=T,
            device=device,
            num_classes=num_classes,
        )

        loss = F.cross_entropy(logits, labels, reduction="sum")
        pred_labels = logits.argmax(dim=-1)

        total_loss += loss.item()
        total_correct += (pred_labels == labels).sum().item()
        total_examples += labels.numel()

    # Restore state.
    experiment.prior = original_prior

    if deterministic_observations and original_noise_scale is not None:
        experiment.noise_scale = original_noise_scale

    return {
        f"{split}/ce_loss": total_loss / total_examples,
        f"{split}/accuracy": total_correct / total_examples,
        f"{split}/n_examples": total_examples,
    }


def load_model_weights(model, checkpoint_path: str, device: torch.device):
    """
    Robust loader for either:
        - raw state_dict saved by save_state_dict(model, ...)
        - checkpoint dict containing model_state_dict / state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Might already be a raw state_dict.
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    return model


def build_model_and_task(cfg, device):
    experiment = hydra.utils.instantiate(cfg.task)

    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)

    model = Aline(embedder, encoder, head).to(device)

    return model, experiment


@torch.no_grad()
def collect_design_trajectories(
    cfg,
    model,
    experiment,
    batch_size,
    T,
    device,
    deterministic_observations=False,
):
    """
    Roll out the learned design policy and collect selected designs.

    Returns
    -------
    out : dict
        Contains selected_x: [B, T, dim_x]
                 selected_idx: [B, T]
                 target_all: task-dependent target
                 final_batch: final ALINE batch
    """

    model.eval()

    batch = experiment.sample_batch(batch_size)

    # Move tensors to device if needed
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    selected_x = []
    selected_idx = []
    log_probs = []

    for t in range(T):
        if getattr(cfg, "time_token", False):
            batch.t = torch.tensor([t / T], device=device)

        pred = model.forward(batch)
        design_out = pred.design_out

        idx = design_out.idx

        # idx can be shape [B] or [B, 1]
        idx_flat = idx.squeeze(-1).long()

        # query_x is usually [B, n_query, dim_x]
        chosen_x = batch.query_x[
            torch.arange(batch.query_x.shape[0], device=device),
            idx_flat,
        ]

        selected_x.append(chosen_x.detach().cpu())
        selected_idx.append(idx_flat.detach().cpu())

        if hasattr(design_out, "log_prob"):
            log_probs.append(design_out.log_prob.detach().cpu())

        batch = experiment.update_batch(batch, idx)

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

    selected_x = torch.stack(selected_x, dim=1)      # [B, T, dim_x]
    selected_idx = torch.stack(selected_idx, dim=1)  # [B, T]

    out = {
        "selected_x": selected_x,
        "selected_idx": selected_idx,
        "target_all": batch.target_all.detach().cpu() if hasattr(batch, "target_all") else None,
        "final_batch": batch,
    }

    if len(log_probs) > 0:
        out["log_probs"] = torch.stack(log_probs, dim=1)

    return out


def _design_to_pixel_coords(x, image_shape=(28, 28)):
    """
    Convert a design x to pixel coordinates.

    Supports:
    - dim_x = 1: flattened pixel index, either integer-like or normalized in [0, 1]
    - dim_x = 2: coordinates, either pixel-space or normalized in [0, 1]
    """

    H, W = image_shape
    x = np.asarray(x)

    if x.shape[-1] == 1:
        val = float(x[0])

        # Normalized scalar in [0, 1]
        if 0.0 <= val <= 1.0:
            idx = int(round(val * (H * W - 1)))
        else:
            idx = int(round(val))

        idx = np.clip(idx, 0, H * W - 1)
        row = idx // W
        col = idx % W
        return row, col

    elif x.shape[-1] == 2:
        a, b = float(x[0]), float(x[1])

        # Normalized coordinates in [0, 1]
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            row = int(round(a * (H - 1)))
            col = int(round(b * (W - 1)))
        else:
            row = int(round(a))
            col = int(round(b))

        row = np.clip(row, 0, H - 1)
        col = np.clip(col, 0, W - 1)
        return row, col

    else:
        raise ValueError(f"Cannot visualize design with dim_x={x.shape[-1]}")


def plot_design_trajectories_mnist(
    selected_x,
    target_images=None,
    labels=None,
    image_shape=(28, 28),
    num_examples=8,
    save_path=None,
):
    """
    Plot selected design locations over MNIST images.

    Parameters
    ----------
    selected_x : torch.Tensor or np.ndarray
        Shape [B, T, dim_x].
    target_images : optional
        Shape [B, 1, H, W], [B, H, W], or flattened [B, H*W].
        If None, plots selected designs on a blank canvas.
    labels : optional
        Class labels or target labels.
    """

    selected_x = np.asarray(selected_x)
    B, T, dim_x = selected_x.shape

    num_examples = min(num_examples, B)
    H, W = image_shape

    fig, axes = plt.subplots(
        1,
        num_examples,
        figsize=(2.2 * num_examples, 2.5),
        dpi=160,
        squeeze=False,
    )
    axes = axes[0]

    for n in range(num_examples):
        ax = axes[n]

        if target_images is not None:
            img = target_images[n]

            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()

            img = np.asarray(img)

            if img.ndim == 3:
                img = img.squeeze(0)

            if img.ndim == 1:
                img = img.reshape(H, W)

            ax.imshow(img, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)

        coords = np.array([
            _design_to_pixel_coords(selected_x[n, t], image_shape=image_shape)
            for t in range(T)
        ])

        rows = coords[:, 0]
        cols = coords[:, 1]

        # Plot trajectory
        ax.plot(cols, rows, linewidth=1.2, marker="o", markersize=3)

        # Annotate step numbers
        for t, (r, c) in enumerate(zip(rows, cols)):
            ax.text(
                c + 0.5,
                r + 0.5,
                str(t + 1),
                fontsize=6,
                ha="left",
                va="bottom",
            )

        if labels is not None:
            label = labels[n]
            if torch.is_tensor(label):
                label = label.item()
            ax.set_title(f"label={label}", fontsize=9)
        else:
            ax.set_title(f"example {n}", fontsize=9)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()



@hydra.main(version_base=None, config_path="./config", config_name="eval_mnist")
def main(cfg):
    device = torch.device(cfg.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, experiment = build_model_and_task(cfg, device)

    if not hasattr(cfg, "eval_checkpoint") or cfg.eval_checkpoint is None:
        raise ValueError(
            "Please provide eval_checkpoint=/path/to/model.pth or checkpoint.tar"
        )

    model = load_model_weights(model, cfg.eval_checkpoint, device)

    T_eval = getattr(cfg, "T_eval", cfg.T)
    batch_size_eval = getattr(cfg, "eval_batch_size_classif", 256)
    num_batches_eval = getattr(cfg, "eval_num_batches_classif", 20)
    deterministic_observations = getattr(cfg, "deterministic_observations", False)

    num_classes = cfg.task.n_target_theta

    train_metrics = evaluate_split(
        cfg=cfg,
        model=model,
        experiment=experiment,
        split="train",
        batch_size=batch_size_eval,
        num_batches=num_batches_eval,
        T=T_eval,
        device=device,
        num_classes=num_classes,
        deterministic_observations=deterministic_observations,
    )

    test_metrics = evaluate_split(
        cfg=cfg,
        model=model,
        experiment=experiment,
        split="test",
        batch_size=batch_size_eval,
        num_batches=num_batches_eval,
        T=T_eval,
        device=device,
        num_classes=num_classes,
        deterministic_observations=deterministic_observations,
    )

    print("\n=== Classification evaluation ===")
    for k, v in {**train_metrics, **test_metrics}.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------
    # Optional qualitative visualization of selected designs
    # ------------------------------------------------------------

    if getattr(cfg, "visualize_designs", False):
        print("\n=== Visualizing design choices ===")

        num_examples = getattr(cfg, "num_visualized_examples", 8)
        design_vis_batch_size = getattr(cfg, "design_vis_batch_size", num_examples)

        traj = collect_design_trajectories(
            cfg=cfg,
            model=model,
            experiment=experiment,
            batch_size=design_vis_batch_size,
            T=T_eval,
            device=device,
            deterministic_observations=deterministic_observations,
        )

        selected_x = traj["selected_x"]
        batch = traj["final_batch"]

        target_images = None
        labels = None

        if hasattr(batch, "image"):
            target_images = batch.image.detach().cpu()
        elif hasattr(batch, "images"):
            target_images = batch.images.detach().cpu()
        elif hasattr(batch, "theta_image"):
            target_images = batch.theta_image.detach().cpu()
        elif hasattr(batch, "target_image"):
            target_images = batch.target_image.detach().cpu()

        if hasattr(batch, "label"):
            labels = batch.label.detach().cpu()
        elif hasattr(batch, "labels"):
            labels = batch.labels.detach().cpu()
        elif hasattr(batch, "theta"):
            theta = batch.theta.detach().cpu()
            labels = theta.argmax(dim=-1) if theta.ndim == 2 else theta
        elif hasattr(batch, "target_all"):
            target_all = batch.target_all.detach().cpu()
            labels = target_all.argmax(dim=-1) if target_all.ndim == 2 else target_all

        save_dir = Path(getattr(cfg, "output_dir", ".")) / "design_visualizations"
        save_path = save_dir / "selected_design_trajectories.pdf"

        plot_design_trajectories_mnist(
            selected_x=selected_x,
            target_images=target_images,
            labels=labels,
            image_shape=(28, 28),
            num_examples=num_examples,
            save_path=save_path,
        )

        print(f"Saved design visualization to {save_path}")

        
if __name__ == "__main__":
    main()
    