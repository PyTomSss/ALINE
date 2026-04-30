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


@hydra.main(version_base=None, config_path="./config", config_name="eval_mnist")
def main(cfg):
    device = torch.device(cfg.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, experiment = build_model_and_task(cfg, device)

    #cfg.eval_checkpoint = "/checkpoints/61844_1/model/aae_MaskedMNIST_p5_q128_T5_1.pth"
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


if __name__ == "__main__":
    main()
    