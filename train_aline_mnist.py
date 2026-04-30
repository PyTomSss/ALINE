import torch
import torch.nn.functional as F
import os

from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import random
import numpy as np
import time

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from model import Aline
from utils import (
    create_logger,
    set_seed,
    save_state_dict,
    load_checkpoint,
    save_checkpoint,
    set_layerwise_lr,
    eval_boed,
)


os.environ["HYDRA_FULL_ERROR"] = "1"

def get_class_labels_from_batch(batch):
    """
    Extract integer class labels from an ALINE/MNIST batch.

    Priority:
    - batch.label / batch.labels / batch.target_label if available
    - otherwise infer from batch.target_all
    """

    if hasattr(batch, "label"):
        labels = batch.label
    elif hasattr(batch, "labels"):
        labels = batch.labels
    elif hasattr(batch, "target_label"):
        labels = batch.target_label
    elif hasattr(batch, "theta"):
        labels = batch.theta
    elif hasattr(batch, "target_all"):
        labels = batch.target_all
    else:
        raise AttributeError(
            "Could not find class labels in batch. Expected one of: "
            "`label`, `labels`, `target_label`, `theta`, or `target_all`."
        )

    # If one-hot or class-probability vector: [B, C] -> [B]
    if labels.ndim == 2 and labels.shape[-1] > 1:
        labels = labels.argmax(dim=-1)

    # If [B, 1] -> [B]
    if labels.ndim > 1:
        labels = labels.squeeze(-1)

    return labels.long()


def compute_classification_ll(logits, labels):
    """
    Compute log q_phi(c | H_t) for classification.

    Parameters
    ----------
    logits : torch.Tensor
        Shape [B, C].
    labels : torch.Tensor
        Shape [B].

    Returns
    -------
    ll : torch.Tensor
        Shape [B], log probability assigned to the true class.
    """

    if logits.ndim != 2:
        raise ValueError(
            f"Expected posterior logits with shape [B, C], got {tuple(logits.shape)}."
        )

    labels = labels.long()
    if labels.ndim > 1:
        labels = labels.squeeze(-1)

    log_probs = F.log_softmax(logits, dim=-1)

    ll = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    return ll


def classification_accuracy_from_logits(logits, labels):
    labels = labels.long()
    if labels.ndim > 1:
        labels = labels.squeeze(-1)

    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean()


def train(
    cfg,
    logger,
    model,
    experiment,
    batch_size: int,
    min_T: int,
    max_T: int,
    max_epoch: int,
    verbose: int = 10,
):
    """
    MNIST-specific ALINE training.

    The posterior head is assumed to output categorical logits:

        posterior_out.logits : [B, num_classes]

    The posterior is trained with cross-entropy, and the design policy is trained
    with a REINFORCE-style reward given by reductions in classification NLL.
    """

    optimizer, scheduler = set_layerwise_lr(cfg, model)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    if cfg.load_checkpoint is True:
        start_epoch, optimizer, scheduler = load_checkpoint(
            cfg, model, optimizer, scheduler, cfg.load_path
        )
    else:
        start_epoch = 0

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    # During burning, use a smaller candidate query set.
    if start_epoch < cfg.burning_epoch:
        experiment.n_query_init = cfg.T

    losses = []
    training_times = []

    device = torch.device(cfg.device)

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()

        model.train()
        optimizer.zero_grad()

        T = random.randint(min_T, max_T)
        batch = experiment.sample_batch(batch_size)

        # Move batch tensors to device.
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        labels = get_class_labels_from_batch(batch).to(device)

        # Store log probabilities of selected designs.
        log_probs = torch.zeros((batch_size, T), device=device)

        # Per-example NLLs at each acquisition step.
        nlls_for_prediction = []
        nlls_for_query = []

        # Optional logging.
        accuracies = []
        confidences = []

        for t in range(T):
            if cfg.time_token:
                batch.t = torch.tensor([t / T], device=device)

            pred = model.forward(batch)

            design_out = pred.design_out
            posterior_out = pred.posterior_out

            if not hasattr(posterior_out, "logits"):
                raise AttributeError(
                    "For MNIST classification, `posterior_out` must contain "
                    "`logits` with shape [B, num_classes]. "
                    "You probably need to modify the posterior head."
                )

            logits = posterior_out.logits

            # Classification posterior log-likelihood.
            target_ll = compute_classification_ll(
                logits=logits,
                labels=labels,
            )  # [B]

            nll = -target_ll  # [B]

            nlls_for_query.append(nll)
            nlls_for_prediction.append(nll)

            with torch.no_grad():
                acc_t = classification_accuracy_from_logits(logits, labels)
                conf_t = torch.softmax(logits, dim=-1).max(dim=-1).values.mean()
                accuracies.append(acc_t.detach())
                confidences.append(conf_t.detach())

            log_probs[:, t] = design_out.log_prob.to(device)

            # Update the active-learning batch with the chosen query/design.
            batch = experiment.update_batch(batch, design_out.idx)

            # Ensure updated tensors are on the correct device.
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

        # Shape [B, T]
        nlls_for_query = torch.stack(nlls_for_query, dim=1)
        nlls_for_prediction = torch.stack(nlls_for_prediction, dim=1)

        # ------------------------------------------------------------
        # REINFORCE reward: improvement in posterior classification NLL
        # ------------------------------------------------------------

        rewards = []

        for t in range(1, T):
            nll_gain = torch.clamp(
                nlls_for_query[:, t - 1] - nlls_for_query[:, t],
                min=0.0,
            ).detach()  # [B]

            reward = (cfg.gamma ** t) * nll_gain
            rewards.append(reward)

        if len(rewards) > 0:
            R = torch.stack(rewards, dim=1)  # [B, T-1]

            # Normalize reward per time step.
            R = (R - R.mean(dim=0, keepdim=True)) / (
                R.std(dim=0, keepdim=True) + 1e-9
            )

            design_loss = -torch.mean(log_probs[:, :-1] * R)
        else:
            design_loss = torch.tensor(0.0, device=device)

        predict_loss = nlls_for_prediction.mean()

        if epoch < cfg.burning_epoch:
            loss = predict_loss
        else:
            loss = cfg.alpha * design_loss + predict_loss

        loss.backward()

        losses.append(loss.item())

        if cfg.clip_grads:
            clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                norm_type="inf",
            )

        # Switch learning-rate configuration after burning.
        if epoch == cfg.burning_epoch:
            optimizer, scheduler = set_layerwise_lr(cfg, model, epoch)

            experiment.n_query_init = cfg.task.n_query_init

            burning_path = save_state_dict(
                model,
                cfg.output_dir,
                f'{cfg.file_name.split(".")[0]}_burning.pth',
            )
            logger.info(f"Model has been saved at {burning_path}")

        optimizer.step()
        scheduler.step()

        end_time = time.time()
        training_times.append(end_time - start_time)

        mean_acc = torch.stack(accuracies).mean()
        final_acc = accuracies[-1]
        mean_conf = torch.stack(confidences).mean()
        final_conf = confidences[-1]

        if cfg.wandb.use_wandb and (epoch % 50 == 0):
            wandb.log(
                {
                    "loss": loss.detach().item(),
                    "classification_nll": predict_loss.detach().item(),
                    "likelihood": -predict_loss.detach().item(),
                    "design_loss": design_loss.detach().item(),
                    "mean_accuracy": mean_acc.detach().item(),
                    "final_accuracy": final_acc.detach().item(),
                    "mean_confidence": mean_conf.detach().item(),
                    "final_confidence": final_conf.detach().item(),
                    "T": T,
                },
                step=epoch,
            )

        # Careful: the original condition was `epoch + 1 % verbose == 0`,
        # which is parsed incorrectly. This is the intended condition.
        if (epoch + 1) % verbose == 0:
            logger.info(
                f"Epoch: {epoch + 1}, "
                f"loss: {losses[-1]:.4f}, "
                f"T: {T}, "
                f"classification_nll: {predict_loss.detach().item():.4f}, "
                f"design_loss: {design_loss.detach().item():.4f}, "
                f"mean_acc: {mean_acc.detach().item():.4f}, "
                f"final_acc: {final_acc.detach().item():.4f}, "
                f"mean_conf: {mean_conf.detach().item():.4f}, "
                f"final_conf: {final_conf.detach().item():.4f}"
            )

            # EIG evaluation may still assume continuous likelihoods depending
            # on eval_boed implementation. Keep it only if already adapted.
            if cfg.eval.EIG:
                bounds = eval_boed(
                    model,
                    experiment,
                    cfg.T - cfg.task.n_context_init,
                    cfg.eval.L,
                    cfg.eval.M,
                    cfg.eval.batch_size,
                    cfg.time_token,
                    False,
                )
                pce_loss = bounds["pce_mean"]
                nmc_loss = bounds["nmc_mean"]

                logger.info(f"PCE: {pce_loss}\tNMC: {nmc_loss}")

                if cfg.wandb.use_wandb:
                    wandb.log(
                        {
                            "PCE": pce_loss,
                            "NMC": nmc_loss,
                        },
                        step=epoch,
                    )

        next_epoch = epoch + 1

        if cfg.checkpoint and next_epoch % cfg.checkpoint == 0:
            save_checkpoint(
                cfg,
                model,
                optimizer,
                scheduler,
                next_epoch,
                with_epoch=True,
            )

        save_every = getattr(cfg, "save_every_after_burning", None)

        if save_every is not None and save_every > 0:
            if (
                next_epoch >= cfg.burning_epoch
                and (next_epoch - cfg.burning_epoch) % save_every == 0
            ):
                ckpt_dir = os.path.join(
                    cfg.output_dir,
                    "checkpoints_after_burning",
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                weight_path = save_state_dict(
                    model,
                    ckpt_dir,
                    f"{cfg.file_name.split('.')[0]}_epoch_{next_epoch}.pth",
                )

                save_checkpoint(
                    cfg,
                    model,
                    optimizer,
                    scheduler,
                    next_epoch,
                    with_epoch=True,
                )

                logger.info(f"Saved periodic model weights at {weight_path}")

    total_time = sum(training_times)

    if len(training_times[cfg.burning_epoch:]) > 0:
        average_time = np.mean(training_times[cfg.burning_epoch:])
        std_time = np.std(training_times[cfg.burning_epoch:])
    else:
        average_time = np.mean(training_times)
        std_time = np.std(training_times)

    logger.info(
        f"Total training time: {total_time:.2f} seconds "
        f"({total_time / 3600:.2f} hours), "
        f"average time per epoch: {average_time:.2f}+-{std_time:.2f} seconds"
    )

    if cfg.wandb.use_wandb:
        wandb.log(
            {
                "training_time": total_time,
                "training_time_hours": total_time / 3600,
            },
            step=max_epoch,
        )


@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    # ------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    if cfg.fix_seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    if cfg.min_T > cfg.T:
        cfg.min_T = cfg.T

    # ------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------

    logger = create_logger(
        os.path.join(cfg.output_dir, "logs"),
        name=cfg.task.name,
    )

    logger.info("Running MNIST classification training.")
    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))

    # ------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(
                cfg,
                resolve=True,
                throw_on_missing=True,
            ),
            dir=cfg.output_dir,
        )

        try:
            hydra_log_dir = os.path.join(
                HydraConfig.get().runtime.output_dir,
                ".hydra",
            )
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # ------------------------------------------------------------
    # Task / experiment
    # ------------------------------------------------------------

    experiment = hydra.utils.instantiate(cfg.task)
    logger.info(f"Task: {experiment}")

    # MNIST classification sanity checks.
    if not hasattr(cfg.task, "n_target_theta"):
        raise ValueError(
            "For MNIST classification, cfg.task.n_target_theta must be defined "
            "and should correspond to the number of classes."
        )

    num_classes = int(cfg.task.n_target_theta)

    logger.info(f"Number of classes: {num_classes}")

    if num_classes <= 1:
        raise ValueError(
            f"Invalid number of classes: cfg.task.n_target_theta={num_classes}."
        )

    # Optional HPO block kept for compatibility, but normally irrelevant for MNIST.
    if hasattr(experiment, "meta_dataset"):
        logger.info(f"Using HPO-B meta-dataset: {experiment.meta_dataset}")
        logger.info(f"Input dimension: {experiment.dim_x}")
        logger.info(f"Number of datasets: {experiment.hpob.n_dataset}")

        if cfg.task.dim_x != experiment.dim_x:
            logger.info(
                f"Updating dim_x from config value {cfg.task.dim_x} "
                f"to actual dataset dimension {experiment.dim_x}"
            )
            cfg.task.dim_x = experiment.dim_x

        if cfg.task.dim_y != experiment.dim_y:
            logger.info(
                f"Updating dim_y from config value {cfg.task.dim_y} "
                f"to actual dataset dimension {experiment.dim_y}"
            )
            cfg.task.dim_y = experiment.dim_y

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)

    model = Aline(embedder, encoder, head)
    model = model.to(device)

    logger.info(model)
    logger.info(f"Model device: {next(model.parameters()).device}")

    # Optional sanity check on a small batch before training.
    # This will fail early if the posterior head has not yet been adapted
    # to return `posterior_out.logits`.
    with torch.no_grad():
        sanity_batch = experiment.sample_batch(2)

        for k, v in sanity_batch.items():
            if torch.is_tensor(v):
                sanity_batch[k] = v.to(device)

        if getattr(cfg, "time_token", False):
            sanity_batch.t = torch.tensor([0.0], device=device)

        sanity_pred = model.forward(sanity_batch)

        if not hasattr(sanity_pred.posterior_out, "logits"):
            raise AttributeError(
                "MNIST classification training expects "
                "`pred.posterior_out.logits` with shape [B, num_classes]. "
                "Please modify cfg.head / the posterior head accordingly."
            )

        logits = sanity_pred.posterior_out.logits

        if logits.ndim != 2:
            raise ValueError(
                f"Expected posterior logits with shape [B, C], "
                f"got {tuple(logits.shape)}."
            )

        if logits.shape[-1] != num_classes:
            raise ValueError(
                f"Expected logits.shape[-1] == cfg.task.n_target_theta "
                f"({num_classes}), got {logits.shape[-1]}."
            )

        logger.info(
            f"Sanity check passed: posterior logits shape = {tuple(logits.shape)}"
        )

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    train(
        cfg=cfg,
        logger=logger,
        model=model,
        experiment=experiment,
        batch_size=cfg.batch_size,
        min_T=cfg.min_T,
        max_T=cfg.T,
        max_epoch=cfg.max_epoch,
        verbose=cfg.verbose,
    )

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------

    final_path = save_state_dict(
        model,
        cfg.output_dir,
        cfg.file_name,
    )

    logger.info(f"Model has been saved at {final_path}")

    # ------------------------------------------------------------
    # Optional EIG evaluation
    # ------------------------------------------------------------
    # Warning:
    # eval_boed may still assume continuous likelihoods / mixture density
    # posterior. Keep this only if eval_boed has been adapted to the
    # classification posterior.

    if cfg.eval.EIG:
        logger.warning(
            "cfg.eval.EIG=True. Make sure eval_boed has been adapted to "
            "classification logits. Otherwise this evaluation may be invalid."
        )

        experiment.n_query_init = cfg.eval.n_query_final

        bounds = eval_boed(
            model,
            experiment,
            cfg.eval.T_final - cfg.task.n_context_init,
            cfg.eval.L_final,
            cfg.eval.M_final,
            cfg.eval.batch_size_final,
            cfg.time_token,
            stepwise=True,
        )

        logger.info(bounds)

        try:
            logger.info(
                f"PCE: {bounds['pce_mean'][cfg.T - 1]:.3f}"
                f"+-{bounds['pce_err'][cfg.T - 1]:.3f}\t"
                f"NMC: {bounds['nmc_mean'][cfg.T - 1]:.3f}"
                f"+-{bounds['nmc_err'][cfg.T - 1]:.3f}"
            )
        except Exception:
            logger.info(
                f"PCE: {bounds.get('pce_mean', None)}\t"
                f"NMC: {bounds.get('nmc_mean', None)}"
            )

        save_path = os.path.join(
            cfg.output_dir,
            "eval",
            f"{cfg.file_name.split('.')[0]}_"
            f"N{cfg.eval.n_query_final}_T{cfg.eval.T_final}.tar",
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(bounds, save_path)

        logger.info(f"Bounds have been saved at {save_path}.")

    # ------------------------------------------------------------
    # Finish W&B
    # ------------------------------------------------------------

    if cfg.wandb.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
