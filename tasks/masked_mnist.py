import math
from typing import Optional, Union, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions import Distribution

import torchvision
import torchvision.transforms as transforms

from tasks.base_task import Task


class MNISTDatasetPrior(Distribution, torch.nn.Module):
    """
    Prior empirique uniforme sur les images d'un dataset MNIST-like.

    Chaque sample retourne une image theta tirée uniformément parmi les éléments du dataset.

    Si return_labels=True, retourne aussi les labels associés.
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        dataset,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        return_labels: bool = False,
    ) -> None:
        super().__init__()

        self.return_labels = return_labels

        self.register_buffer(
            "_dummy",
            torch.empty(0, device=device, dtype=dtype)
        )

        if hasattr(dataset, "name"): 

            if dataset.name == "cifar": 
                ## Voir comment traiter les masks pour les images en couleur
                self.dataset = dataset.dataset

                self.images = (
                    torch.from_numpy(self.dataset.data)
                    .permute(0, 3, 1, 2)   # NHWC -> NCHW
                    .to(dtype=self.dtype) / 255.0
                )

                self.labels = torch.tensor(self.dataset.targets, dtype=torch.long)
            
            elif dataset.name == "mnist":
                self.dataset = dataset.dataset

                self.images = self.dataset.data.unsqueeze(1).to(dtype=self.dtype) / 255.0
                self.labels = self.dataset.targets.long()

            elif dataset.name == "cifar100": 
                self.dataset = dataset.dataset

                self.images = (
                    torch.from_numpy(self.dataset.data)
                    .permute(0, 3, 1, 2)   # NHWC -> NCHW
                    .to(dtype=self.dtype) / 255.0
                )

                self.labels = torch.tensor(self.dataset.targets, dtype=torch.long)

        else: 
            
            self.images = dataset.data.unsqueeze(1).to(dtype=self.dtype) / 255.0
            self.labels = dataset.targets.long()

        self.n = len(dataset)
            

    @property
    def device(self):
        return self._dummy.device

    @property
    def dtype(self):
        return self._dummy.dtype

    def sample_old(self, n_samples: int = 1):
        """
        Returns:
            thetas: [n_samples, 1, 28, 28]
        or if return_labels=True:
            (thetas, labels)
                thetas: [n_samples, 1, 28, 28]
                labels: [n_samples]
        """
        indices = torch.randint(
            low=0,
            high=self.n,
            size=(n_samples,),
            device=self.device,
        )

        images = []
        labels = []

        for idx in indices.tolist():
            image, label = self.dataset[idx]

            if not torch.is_tensor(image):
                image = torch.tensor(image, dtype=self.dtype)

            image = image.to(device=self.device, dtype=self.dtype)
            images.append(image)

            if self.return_labels:
                labels.append(label)

        thetas = torch.stack(images, dim=0)

        if self.return_labels:
            labels = torch.tensor(labels, device=self.device, dtype=torch.long)
            return thetas, labels

        return thetas
    
    def sample(self, n_samples: int = 1):
        indices = torch.randint(
            low=0,
            high=self.n,
            size=(n_samples,),
            device=self.images.device,
        )

        thetas = self.images[indices].to(self.device, dtype=self.dtype, non_blocking=True)

        if self.return_labels:
            labels = self.labels[indices].to(self.device, non_blocking=True)
            return thetas, labels

        return thetas

    ## PAs testé encore ici ->
    def sample_class(
        self,
        n_samples: int = 1,
        class_id: int = None,
        class_ids: torch.Tensor = None,
    ):
        """
        Sample images from the dataset, optionally restricted to specific class(es).

        Args:
            n_samples: number of samples
            class_id: single class to sample from (int)
            class_ids: tensor/list of allowed classes

        Returns:
            thetas or (thetas, labels)
        """

        if class_id is not None and class_ids is not None:
            raise ValueError("Specify either class_id or class_ids, not both.")

        # ------------------------------------------------------------------
        # Build mask of allowed indices
        # ------------------------------------------------------------------
        if class_id is not None:
            mask = (self.labels == class_id)

        elif class_ids is not None:
            if not torch.is_tensor(class_ids):
                class_ids = torch.tensor(class_ids, device=self.labels.device)
            mask = torch.isin(self.labels, class_ids)

        else:
            mask = None

        # ------------------------------------------------------------------
        # Sample indices
        # ------------------------------------------------------------------
        if mask is None:
            # standard sampling
            indices = torch.randint(
                low=0,
                high=self.n,
                size=(n_samples,),
                device=self.images.device,
            )
        else:
            valid_indices = torch.where(mask)[0]

            if len(valid_indices) == 0:
                raise ValueError("No samples found for the requested class(es).")

            rand_idx = torch.randint(
                low=0,
                high=valid_indices.shape[0],
                size=(n_samples,),
                device=self.images.device,
            )
            indices = valid_indices[rand_idx]

        # ------------------------------------------------------------------
        # Fetch data
        # ------------------------------------------------------------------
        thetas = self.images[indices].to(
            self.device, dtype=self.dtype, non_blocking=True
        )

        if self.return_labels:
            labels = self.labels[indices].to(self.device, non_blocking=True)
            return thetas, labels

        return thetas

    def log_prob(self, theta: Any) -> torch.Tensor: ...


class AttrDict(dict):
    """Dict with attribute access, compatible with ALINE-style batch objects."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class MaskedMNISTClassification(Task):
    """
    ALINE-compatible masked MNIST classification task.

    Generative interpretation
    -------------------------
    theta  : MNIST image [B, C, H, W]
    xi     : top-left corner of a patch, [B, 2], with coordinates (row, col)
    y      : noisy observed patch, flattened, [B, C * patch_size * patch_size]

    ALINE interpretation
    --------------------
    query_x    : candidate patch corners [B, n_query_init, 2]
    context_x  : previously selected patch corners [B, t, 2]
    context_y  : observed noisy patches [B, t, C * p * p]
    target_all : one-hot digit label [B, num_classes]

    Padding is allowed by default:
        row, col in [-(p - 1), H - 1]
    so a patch may lie partially outside the image. Outside-image values are
    handled by grid_sample padding_mode, usually "zeros".
    """

    def __init__(
        self,
        # ALINE-required / config fields
        name: str = "MaskedMNIST",
        embedding_type: str = "theta",
        mask_type: Optional[list[str]] = None,
        n_target_data: int = 0,
        n_target_theta: int = 10,
        n_selected_targets: Optional[int] = None,
        predefined_masks=None,
        predefined_mask_weights=None,
        mask_index=None,
        attend_to=None,
        n_context_init: int = 1,
        n_query_init: int = 128,

        # Task parameters
        patch_size: int = 5,
        noise_scale: float = 0.05,
        dim_theta: int = 28 * 28,
        dim_x: int = 2,
        dim_xi : int = 2, 
        align_corners: bool = True,
        padding_mode: str = "zeros",
        allow_padding: bool = True,
        reparameterize_design: bool = False,
        num_classes: int = 10,
        in_channels: int = 1,
        design_input_range: str = "pixel",

        # Data
        dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        prior: Optional[Distribution] = None,
        prior_test: Optional[Distribution] = None,
        data_root: str = "./data",
        download: bool = False,

        # Runtime
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        _seed: Optional[int] = None,

        # Compatibilité avec ALINE
        dim_y: int = 25, 
    ) -> None:

        self.name = name
        self.embedding_type = embedding_type
        self.mask_type = ["all"] if mask_type is None else mask_type
        self.n_target_data = n_target_data
        self.n_target_theta = n_target_theta
        self.n_selected_targets = n_selected_targets
        self.predefined_masks = predefined_masks
        self.predefined_mask_weights = predefined_mask_weights
        self.mask_index = mask_index
        self.attend_to = attend_to

        self.n_context_init = n_context_init
        self.n_query_init = n_query_init

        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.dim_theta = dim_theta
        self.dim_xi = dim_xi
        self.dim_x = dim_x
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.allow_padding = allow_padding
        self.reparameterize_design = reparameterize_design
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.design_input_range = design_input_range

        self.dtype = dtype
        self.dim_y = in_channels * patch_size * patch_size

        if dataset is None:
            self.dataset = torchvision.datasets.MNIST(
                root=data_root,
                train=True,
                download=download,
                transform=transforms.ToTensor(),
            )
        else:
            self.dataset = dataset

        if test_dataset is None:
            self.test_dataset = torchvision.datasets.MNIST(
                root=data_root,
                train=False,
                download=download,
                transform=transforms.ToTensor(),
            )
        else:
            self.test_dataset = test_dataset

        if prior is None:
            self.prior = MNISTDatasetPrior(
                self.dataset,
                device=device,
                dtype=dtype,
                return_labels=True,
            )

        #super().__init__(prior=prior, device=device)
        super().__init__(device=device)

        if prior_test is None:
            self.prior_test = MNISTDatasetPrior(
                self.test_dataset,
                device=device,
                dtype=dtype,
                return_labels=True,
            )
        else:
            self.prior_test = prior_test

        self.generator = torch.Generator(device=self.device)
        if _seed is not None:
            self.generator.manual_seed(_seed)

        p = self.patch_size
        gy, gx = torch.meshgrid(
            torch.arange(p, dtype=torch.float32),
            torch.arange(p, dtype=torch.float32),
            indexing="ij",
        )

        # If Task inherits from nn.Module, this is the cleanest version.
        self.register_buffer("local_y", gy, persistent=False)
        self.register_buffer("local_x", gx, persistent=False)

    def __repr__(self) -> str:
        return (
            f"MaskedMNISTClassification("
            f"patch_size={self.patch_size}, "
            f"noise_scale={self.noise_scale}, "
            f"allow_padding={self.allow_padding}, "
            f"n_query_init={self.n_query_init})"
        )

    # =====================================================================================
    # ALINE API
    # =====================================================================================

    def sample_batch(self, batch_size: int):
        """
        Sample an ALINE-compatible batch.

        Returns
        -------
        batch.context_x:
            [B, n_context_init, 2]

        batch.context_y:
            [B, n_context_init, C*p*p]

        batch.query_x:
            [B, n_query_init, 2]

        batch.target_all:
            [B, num_classes]
        """
        theta, labels = self.prior.sample(batch_size)

        theta = theta.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device)

        if theta.ndim == 3:
            theta = theta.unsqueeze(1)

        B, C, H, W = theta.shape

        all_corners = self._make_all_patch_corners(H, W)  # [N, 2]
        n_all = all_corners.shape[0]

        observed_mask = torch.zeros(B, n_all, device=self.device, dtype=torch.bool)

        # Initial context: reveal n_context_init random patches.
        if self.n_context_init > 0:
            init_indices = self._sample_candidate_indices(
                observed_mask=observed_mask,
                n_candidates=self.n_context_init,
            )  # [B, n_context_init]

            init_designs = all_corners[init_indices]  # [B, n_context_init, 2]

            eps = torch.randn(
                B,
                self.n_context_init,
                self.dim_y,
                device=self.device,
                dtype=self.dtype,
                generator=self.generator,
            )

            init_y, _, used_designs = self.reparam_make_outcomes(
                theta=theta,
                designs=init_designs,
                eps=eps,
                reparameterize_design=False,
            )

            context_x = used_designs
            context_y = init_y

            arange = torch.arange(B, device=self.device)[:, None]
            observed_mask[arange, init_indices] = True

        else:
            context_x = torch.empty(B, 0, self.dim_xi, device=self.device, dtype=self.dtype)
            context_y = torch.empty(B, 0, self.dim_y, device=self.device, dtype=self.dtype)

        query_indices = self._sample_candidate_indices(
            observed_mask=observed_mask,
            n_candidates=self.n_query_init,
        )  # [B, n_query_init]

        query_x = all_corners[query_indices]  # [B, n_query_init, 2]

        target_all = self._one_hot_labels(labels)  # [B, num_classes]

        batch = AttrDict(
            theta=theta,
            labels=labels,
            target_all=target_all,

            context_x=context_x,
            context_y=context_y,

            query_x=query_x,
            query_indices=query_indices,
            all_corners=all_corners,
            observed_mask=observed_mask,
        )

        return batch

    def update_batch(self, batch, idx: Tensor):
        """
        Update batch after ALINE selects one candidate patch.

        Parameters
        ----------
        batch:
            ALINE batch.

        idx:
            Index into batch.query_x. Shape [B], [B, 1], or compatible.

        Returns
        -------
        batch:
            Updated batch with selected patch appended to context.
        """
        if idx.ndim == 2:
            idx = idx.squeeze(-1)

        idx = idx.long().to(self.device)

        B = batch.theta.shape[0]
        arange = torch.arange(B, device=self.device)

        selected_designs = batch.query_x[arange, idx]  # [B, 2]

        y = self._sample_outcome(
            theta=batch.theta,
            design=selected_designs,
            reparameterize_design=False,
        )  # [B, dim_y]

        batch.context_x = torch.cat(
            [batch.context_x, selected_designs.unsqueeze(1)],
            dim=1,
        )

        batch.context_y = torch.cat(
            [batch.context_y, y.unsqueeze(1)],
            dim=1,
        )

        selected_global_indices = batch.query_indices[arange, idx]
        batch.observed_mask[arange, selected_global_indices] = True

        query_indices = self._sample_candidate_indices(
            observed_mask=batch.observed_mask,
            n_candidates=self.n_query_init,
        )

        batch.query_indices = query_indices
        batch.query_x = batch.all_corners[query_indices]

        return batch

    # =====================================================================================
    # Candidate patch corners
    # =====================================================================================

    def _make_all_patch_corners(self, H: int, W: int) -> Tensor:
        """
        Construct all possible top-left corners.

        If allow_padding=True:
            row in [-(p-1), H-1]
            col in [-(p-1), W-1]

            For MNIST H=W=28, p=5:
                row, col in [-4, ..., 27]
                number of candidates = 32 * 32 = 1024

        If allow_padding=False:
            row in [0, H-p]
            col in [0, W-p]

            For MNIST H=W=28, p=5:
                row, col in [0, ..., 23]
                number of candidates = 24 * 24 = 576
        """
        p = self.patch_size

        if self.allow_padding:
            row_min, row_max = -(p - 1), H - 1
            col_min, col_max = -(p - 1), W - 1
        else:
            row_min, row_max = 0, H - p
            col_min, col_max = 0, W - p

        if row_max < row_min or col_max < col_min:
            raise ValueError(
                f"Invalid patch range for H={H}, W={W}, patch_size={p}, "
                f"allow_padding={self.allow_padding}."
            )

        rows = torch.arange(row_min, row_max + 1, device=self.device, dtype=self.dtype)
        cols = torch.arange(col_min, col_max + 1, device=self.device, dtype=self.dtype)

        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        corners = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=-1)

        return corners  # [N, 2]

    def _sample_candidate_indices(self, observed_mask: Tensor, n_candidates: int) -> Tensor:
        """
        Sample candidate indices among unobserved corners.

        observed_mask:
            [B, N]

        returns:
            [B, n_candidates]
        """
        B, N = observed_mask.shape
        out = []

        for b in range(B):
            available = torch.nonzero(~observed_mask[b], as_tuple=False).squeeze(-1)

            if available.numel() >= n_candidates:
                perm = torch.randperm(
                    available.numel(),
                    device=self.device,
                    generator=self.generator,
                )[:n_candidates]
                chosen = available[perm]

            elif available.numel() > 0:
                extra = available[
                    torch.randint(
                        low=0,
                        high=available.numel(),
                        size=(n_candidates - available.numel(),),
                        device=self.device,
                        generator=self.generator,
                    )
                ]
                chosen = torch.cat([available, extra], dim=0)

            else:
                chosen = torch.randint(
                    low=0,
                    high=N,
                    size=(n_candidates,),
                    device=self.device,
                    generator=self.generator,
                )

            out.append(chosen)

        return torch.stack(out, dim=0)

    def _one_hot_labels(self, labels: Tensor) -> Tensor:
        return F.one_hot(labels.long(), num_classes=self.num_classes).to(
            device=self.device,
            dtype=self.dtype,
        )

    # =====================================================================================
    # Patch extraction
    # =====================================================================================

    def _pixel_to_normalized(self, coord: Tensor, size: int) -> Tensor:
        """
        Convert pixel coordinates to normalized grid_sample coordinates.

        coord:
            Arbitrary shape, pixel coordinates.

        returns:
            Same shape, normalized coordinates in [-1, 1].
            Values outside [-1, 1] are allowed and handled by padding_mode.
        """
        if self.align_corners:
            return 2.0 * coord / (size - 1) - 1.0
        else:
            return 2.0 * (coord + 0.5) / size - 1.0

    def _normalize_designs_to_valid_range(
        self,
        designs: Tensor,
        H: int,
        W: int,
        allow_padding: Optional[bool] = None,
        input_range: Optional[str] = None,
    ) -> Tensor:
        """
        Optional differentiable reparameterization.

        For ALINE discrete candidates, this should NOT be used.
        ALINE passes pixel-space designs directly and uses reparameterize_design=False.

        designs:
            [B, T, 2]

        input_range:
            "unconstrained": maps R to valid interval via sigmoid
            "tanh" / "minus1_1" / "[-1,1]": maps [-1, 1] to valid interval
            "pixel": returns designs unchanged
        """
        if allow_padding is None:
            allow_padding = self.allow_padding

        if input_range is None:
            input_range = self.design_input_range

        if input_range == "pixel":
            return designs

        p = self.patch_size

        if allow_padding:
            row_min, row_max = -(p - 1), H - 1
            col_min, col_max = -(p - 1), W - 1
        else:
            row_min, row_max = 0, H - p
            col_min, col_max = 0, W - p

        row_min_t = torch.as_tensor(row_min, device=designs.device, dtype=designs.dtype)
        row_max_t = torch.as_tensor(row_max, device=designs.device, dtype=designs.dtype)
        col_min_t = torch.as_tensor(col_min, device=designs.device, dtype=designs.dtype)
        col_max_t = torch.as_tensor(col_max, device=designs.device, dtype=designs.dtype)

        if input_range == "unconstrained":
            row = row_min_t + torch.sigmoid(designs[..., 0]) * (row_max_t - row_min_t)
            col = col_min_t + torch.sigmoid(designs[..., 1]) * (col_max_t - col_min_t)

        elif input_range in {"tanh", "minus1_1", "[-1,1]"}:
            row = row_min_t + 0.5 * (designs[..., 0] + 1.0) * (row_max_t - row_min_t)
            col = col_min_t + 0.5 * (designs[..., 1] + 1.0) * (col_max_t - col_min_t)

        else:
            raise ValueError(
                f"Unknown input_range='{input_range}'. "
                "Expected 'pixel', 'unconstrained', or 'tanh'."
            )

        return torch.stack((row, col), dim=-1)

    def _extract_patch_mean(
        self,
        theta: Tensor,
        designs: Tensor,
        reparameterize_design: bool = False,
    ):
        """
        Extract one patch per image.

        theta:
            [B, C, H, W]

        designs:
            [B, 2], top-left corners in pixel coordinates if reparameterize_design=False.

        returns:
            patches: [B, C*p*p]
            used_designs: [B, 2]
        """
        B, C, H, W = theta.shape
        p = self.patch_size
        dtype = theta.dtype
        device = theta.device

        if designs.ndim != 2 or designs.shape[-1] != 2:
            raise ValueError(f"Expected designs of shape [B, 2], got {designs.shape}")

        designs = designs.to(device=device, dtype=dtype)

        if reparameterize_design:
            designs = self._normalize_designs_to_valid_range(
                designs.unsqueeze(1),
                H,
                W,
                allow_padding=self.allow_padding,
                input_range=self.design_input_range,
            ).squeeze(1)

        rows = designs[:, 0].view(B, 1, 1)
        cols = designs[:, 1].view(B, 1, 1)

        local_y = self.local_y.to(device=device, dtype=dtype).view(1, p, p)
        local_x = self.local_x.to(device=device, dtype=dtype).view(1, p, p)

        y_pix = rows + local_y
        x_pix = cols + local_x

        y_grid = self._pixel_to_normalized(y_pix, H)
        x_grid = self._pixel_to_normalized(x_pix, W)

        grid = torch.stack((x_grid, y_grid), dim=-1)  # [B, p, p, 2]

        patches = F.grid_sample(
            theta,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )  # [B, C, p, p]

        patches = patches.reshape(B, C * p * p)
        return patches, designs

    def _extract_patch_mean_batched(
        self,
        theta: Tensor,
        designs: Tensor,
        reparameterize_design: bool = False,
    ):
        """
        Vectorized extraction over time.

        theta:
            [B, C, H, W]

        designs:
            [B, T, 2]

        returns:
            mean_patches: [B, T, C*p*p]
            used_designs: [B, T, 2]
        """
        B, C, H, W = theta.shape
        p = self.patch_size
        dtype = theta.dtype
        device = theta.device

        if designs.ndim != 3 or designs.shape[-1] != 2:
            raise ValueError(f"Expected designs of shape [B, T, 2], got {designs.shape}")

        _, T, _ = designs.shape

        designs = designs.to(device=device, dtype=dtype)

        if reparameterize_design:
            designs = self._normalize_designs_to_valid_range(
                designs,
                H,
                W,
                allow_padding=self.allow_padding,
                input_range=self.design_input_range,
            )

        rows = designs[..., 0].unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        cols = designs[..., 1].unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]

        local_y = self.local_y.to(device=device, dtype=dtype).view(1, 1, p, p)
        local_x = self.local_x.to(device=device, dtype=dtype).view(1, 1, p, p)

        y_pix = rows + local_y  # [B, T, p, p]
        x_pix = cols + local_x  # [B, T, p, p]

        y_grid = self._pixel_to_normalized(y_pix, H)
        x_grid = self._pixel_to_normalized(x_pix, W)

        grid = torch.stack((x_grid, y_grid), dim=-1)  # [B, T, p, p, 2]
        grid = grid.reshape(B * T, p, p, 2)

        theta_bt = (
            theta.unsqueeze(1)
            .expand(B, T, C, H, W)
            .reshape(B * T, C, H, W)
        )

        patches = F.grid_sample(
            theta_bt,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )  # [B*T, C, p, p]

        patches = patches.reshape(B, T, C * p * p)
        return patches, designs

    # =====================================================================================
    # Outcome sampling and likelihood
    # =====================================================================================

    def _sample_outcome(
        self,
        theta: Tensor,
        design: Tensor,
        reparameterize_design: Optional[bool] = None,
    ) -> Tensor:
        """
        Sample one noisy patch.

        theta:
            [B, C, H, W]

        design:
            [B, 2]

        returns:
            y: [B, C*p*p]
        """
        if reparameterize_design is None:
            reparameterize_design = self.reparameterize_design

        mean_patch, _ = self._extract_patch_mean(
            theta=theta,
            designs=design,
            reparameterize_design=reparameterize_design,
        )

        if self.noise_scale > 0:
            mean_patch = mean_patch + self.noise_scale * torch.randn_like(mean_patch)

        return mean_patch

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        reparameterize_design: Optional[bool] = None,
        reduce_time: bool = False,
    ) -> Tensor:
        """
        Log p(y | theta, design) under Gaussian patch noise.

        Single-step:
            y:      [B, C*p*p]
            theta:  [B, C, H, W]
            design: [B, 2]
            returns [B]

        Trajectory:
            y:      [B, T, C*p*p]
            theta:  [B, C, H, W]
            design: [B, T, 2]
            returns [B, T] or [B] if reduce_time=True
        """
        if reparameterize_design is None:
            reparameterize_design = self.reparameterize_design

        if y.ndim not in (2, 3):
            raise ValueError(f"Expected y to have ndim 2 or 3, got shape {y.shape}")

        if design.ndim not in (2, 3):
            raise ValueError(f"Expected design to have ndim 2 or 3, got shape {design.shape}")

        if theta.ndim != 4:
            raise ValueError(f"Expected theta of shape [B, C, H, W], got {theta.shape}")

        single_step = y.ndim == 2 and design.ndim == 2

        if single_step:
            y_ = y.unsqueeze(1)
            design_ = design.unsqueeze(1)
        elif y.ndim == 3 and design.ndim == 3:
            y_ = y
            design_ = design
        else:
            raise ValueError(
                f"Incompatible shapes: y={y.shape}, design={design.shape}."
            )

        mean_patches, _ = self._extract_patch_mean_batched(
            theta=theta,
            designs=design_,
            reparameterize_design=reparameterize_design,
        )

        if y_.shape != mean_patches.shape:
            raise ValueError(
                f"Expected y of shape {mean_patches.shape}, got {y_.shape}"
            )

        if self.noise_scale > 0:
            sigma2 = self.noise_scale ** 2
            log_norm = -0.5 * math.log(2.0 * math.pi * sigma2)
            logp_per_dim = log_norm - 0.5 * (y_ - mean_patches) ** 2 / sigma2
            logp_t = logp_per_dim.sum(dim=-1)
        else:
            exact_match = torch.all(y_ == mean_patches, dim=-1)
            logp_t = torch.where(
                exact_match,
                torch.zeros_like(mean_patches[..., 0]),
                torch.full_like(mean_patches[..., 0], -torch.inf),
            )

        if single_step:
            return logp_t[:, 0]

        return logp_t.sum(dim=1) if reduce_time else logp_t

    # =====================================================================================
    # Contrastive likelihood utilities
    # =====================================================================================

    def _outcome_loc_contrastive(
        self,
        theta_all: Tensor,
        design: Tensor,
    ) -> Tensor:
        """
        Compute patch means for contrastive theta candidates.

        theta_all:
            [B, Lp1, C, H, W]

        design:
            [B, T, 2]

        returns:
            loc: [B, T, Lp1, C*p*p]
        """
        B, Lp1, C, H, W = theta_all.shape
        B2, T, D = design.shape
        p = self.patch_size

        if B2 != B:
            raise ValueError(f"Batch mismatch: theta_all B={B}, design B={B2}.")
        if D != 2:
            raise ValueError(f"Expected design last dim = 2, got {D}")

        design_exp = design.unsqueeze(1).expand(B, Lp1, T, 2)

        theta_flat = theta_all.reshape(B * Lp1, C, H, W)
        design_flat = design_exp.reshape(B * Lp1, T, 2)

        loc_flat, _ = self._extract_patch_mean_batched(
            theta=theta_flat,
            designs=design_flat,
            reparameterize_design=self.reparameterize_design,
        )  # [B*Lp1, T, C*p*p]

        loc_flat = loc_flat.reshape(B, Lp1, T, C * p * p)
        return loc_flat.permute(0, 2, 1, 3)  # [B, T, Lp1, C*p*p]

    def _outcome_logp_contrastive(
        self,
        y: Tensor,
        theta_all: Tensor,
        design: Tensor,
    ) -> Tensor:
        """
        Contrastive log likelihood.

        y:
            [B, T, C*p*p]

        theta_all:
            [B, Lp1, C, H, W]

        design:
            [B, T, 2]

        returns:
            [B, T, Lp1]
        """
        loc = self._outcome_loc_contrastive(theta_all, design)
        y_exp = y.unsqueeze(2)

        if self.noise_scale > 0:
            sigma2 = self.noise_scale ** 2
            log_norm = -0.5 * math.log(2.0 * math.pi * sigma2)
            logp_per_dim = log_norm - 0.5 * (y_exp - loc) ** 2 / sigma2
            return logp_per_dim.sum(dim=-1)

        exact_match = torch.all(y_exp == loc, dim=-1)
        return torch.where(
            exact_match,
            torch.zeros_like(exact_match, dtype=y.dtype),
            torch.full_like(exact_match, -torch.inf, dtype=y.dtype),
        )

    # =====================================================================================
    # Reparameterized outcome generation
    # =====================================================================================

    def _get_num_channels(self) -> int:
        if hasattr(self, "n_channels"):
            return self.n_channels
        if hasattr(self, "in_channels"):
            return self.in_channels
        if hasattr(self, "channels"):
            return self.channels
        return 1

    def reparam_sample_residuals(self, B: int, T: int) -> Tensor:
        """
        Sample residuals for reparameterized rollouts.

        returns:
            eps: [B, T, C*p*p]
        """
        p = self.patch_size
        C = self._get_num_channels()
        dim_y = C * p * p

        return torch.randn(
            B,
            T,
            dim_y,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator,
        )

    def reparam_make_outcome(
        self,
        theta: Tensor,
        designs: Tensor,
        eps: Optional[Tensor] = None,
        designer_mode: bool = True,
        reparameterize_design: Optional[bool] = None,
    ):
        """
        Reparameterized single-step outcome.

        theta:
            [B, C, H, W]

        designs:
            [B, 2]

        eps:
            [B, C*p*p] or None

        returns:
            y if designer_mode=True
            else (y, mean_patches, used_designs)
        """
        if reparameterize_design is None:
            reparameterize_design = self.reparameterize_design

        B, C, _, _ = theta.shape
        p = self.patch_size
        dim_y = C * p * p

        mean_patches, used_designs = self._extract_patch_mean(
            theta=theta,
            designs=designs,
            reparameterize_design=reparameterize_design,
        )

        if eps is None:
            eps = torch.randn(
                B,
                dim_y,
                device=theta.device,
                dtype=theta.dtype,
                generator=self.generator,
            )
        else:
            if eps.shape != mean_patches.shape:
                raise ValueError(
                    f"Expected eps of shape {mean_patches.shape}, got {eps.shape}"
                )
            eps = eps.to(device=theta.device, dtype=theta.dtype)

        if self.noise_scale > 0:
            y = mean_patches + self.noise_scale * eps
        else:
            y = mean_patches

        if designer_mode:
            return y

        return y, mean_patches, used_designs

    def reparam_make_outcomes(
        self,
        theta: Tensor,
        designs: Tensor,
        eps: Optional[Tensor] = None,
        reparameterize_design: Optional[bool] = None,
    ):
        """
        Reparameterized trajectory outcome generation.

        theta:
            [B, C, H, W]

        designs:
            [B, T, 2]

        eps:
            [B, T, C*p*p] or None

        returns:
            y:            [B, T, C*p*p]
            mean_patches: [B, T, C*p*p]
            used_designs: [B, T, 2]
        """
        if reparameterize_design is None:
            reparameterize_design = self.reparameterize_design

        B, C, _, _ = theta.shape
        p = self.patch_size
        dim_y = C * p * p

        mean_patches, used_designs = self._extract_patch_mean_batched(
            theta=theta,
            designs=designs,
            reparameterize_design=reparameterize_design,
        )

        if eps is None:
            eps = torch.randn(
                B,
                mean_patches.shape[1],
                dim_y,
                device=theta.device,
                dtype=theta.dtype,
                generator=self.generator,
            )
        else:
            if eps.shape != mean_patches.shape:
                raise ValueError(
                    f"Expected eps of shape {mean_patches.shape}, got {eps.shape}"
                )
            eps = eps.to(device=theta.device, dtype=theta.dtype)

        if self.noise_scale > 0:
            y = mean_patches + self.noise_scale * eps
        else:
            y = mean_patches

        return y, mean_patches, used_designs