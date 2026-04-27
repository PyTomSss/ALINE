from abc import ABC, abstractmethod

import math
from .base_task import Task
import torch
from torch import nn, Tensor, seed
import torch.distributions as torch_dist
from typing import Optional, Union, Any



def expand_along_batch(*tensors):
    # Expands tensors along the first (batch) dimension so that they agree in this dim.
    # Requires tensors to have either the same batch dimension or a batch dimension of one.
    
    if not tensors:
        return []
    
    batch_sizes = [t.shape[0] for t in tensors]
    target_b = max(batch_sizes)

    for b in batch_sizes:
        # Every batch size must be either 1 or the target batch size.
        if b != 1 and b != target_b:
            raise ValueError(
                f"Incompatible batch dimensions. Batch sizes must either be 1 or "
                f"agree on a single value. Got: {batch_sizes}"
            )
        
    return tuple(t.expand(target_b, *t.shape[1:]) if t.shape[0] == 1 else t for t in tensors)


class Task(ABC):

    def __init__(self, prior, device):
        self.device = torch.device(device) 
        
        self.prior = prior
        self.current_prior = prior
    
    @abstractmethod
    def _outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        returns: [batch, ]
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design, n_samples=1):
        """
        theta, design: [b, **shape]
        returns: [n_samples, b, **shape]
        """
        ...

    def outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        Also supports when any of y, theta, design only have batch=1 by broadcasting.
        """
        assert y.device == self.device, f'Expected y on device {self.device}, but got {y.device}'
        assert theta.device == self.device, f'Expected theta on device {self.device}, but got {theta.device}'
        assert design.device == self.device, f'Expected design on device {self.device}, but got {design.device}'

        y_expanded, theta_expanded, design_expanded = expand_along_batch(y, theta, design)
        return self._outcome_logp(y_expanded, theta_expanded, design_expanded)
    
    def sample_outcome(self, theta, design, n_samples=1):

        assert theta.device == self.device, f'Expected theta on device {self.device}, but got {theta.device}'
        assert design.device == self.device, f'Expected design on device {self.device}, but got {design.device}'

        theta_expanded, design_expanded = expand_along_batch(theta, design)
        return self._sample_outcome(theta_expanded, design_expanded, n_samples=n_samples)
        
    def sample_joint(self, n_samples, design):

        assert design.device == self.device, f'Expected design on device {self.device}, but got {design.device}'
        
        theta = self.sample_current_prior(n_samples)
        outcome = self.sample_outcome(theta, design)
        outcome = outcome.squeeze(0)

        return outcome, theta
    
    def sample_prior(self, n_samples):

        return self.prior.sample(n_samples=n_samples) ## faire attention que la prior sample un tenseur sur le device correct
    
    def sample_current_prior(self, n_samples):
        return self.current_prior.sample(n_samples=n_samples) ## faire attention que la prior sample un tenseur sur le device correct
    
    def update_prior(self, design):
        # Potentially updates the prior
        # For use in sequential experiments
        raise NotImplementedError
    

class Distribution(ABC):
    """
    Abstract class for arbitrary distributions.
    """

    @abstractmethod
    def sample(self, num_samples: int = 1) -> torch.Tensor: ...

    @abstractmethod
    def log_prob(self, theta: Any) -> torch.Tensor: ...


class PriorPendulum(Distribution, nn.Module):
    """
    Prior for pendulum parameters.
    
    family="normal" or "lognormal":
        
    """

    #arg_constraints = {}
    #support = torch_dist.constraints.real  # (lognormal is actually positive; kept generic)

    def __init__(
        self,
        dim_theta: int,
        mu_theta: Optional[Tensor] = None,
        cov_theta: Optional[Tensor] = None,
        misspec_shift: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        family: str = "normal",          
        eps: float = 1e-12,              
    ) -> None:
        
        super().__init__()
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.eps = float(eps)

        if mu_theta is None:
            assert dim_theta == 3
            mu_theta = torch.tensor([14.7, 0.0, 3.0], device=self.device, dtype=self.dtype)

        if cov_theta is None:
            assert dim_theta == 3
            cov_theta = torch.tensor([0.1, 0.01, 0.1], device=self.device, dtype=self.dtype)

        if misspec_shift is None:
            misspec_shift = torch.zeros(dim_theta, device=self.device, dtype=self.dtype)

        if family not in ("normal", "lognormal"):
            raise ValueError(f"family must be 'normal' or 'lognormal', got {family}")

        self.dim_theta = int(dim_theta)
        self.family = family

        self.register_buffer("mu_theta", mu_theta.to(self.device, self.dtype))
        self.register_buffer("cov_theta", cov_theta.to(self.device, self.dtype))
        self.register_buffer("misspec_shift", misspec_shift.to(self.device, self.dtype))

        # Base Gaussian (either on theta directly, or on log(theta))
        self.base = torch_dist.MultivariateNormal(
            loc=self.mu_theta + self.misspec_shift,
            covariance_matrix=torch.diag(self.cov_theta),
        )

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Returns: [n_samples, dim_theta]
        """
        z = self.base.sample((n_samples,))  
        if self.family == "normal":
            return z
        
        # lognormal
        return torch.exp(z)

    def log_prob(self, theta: Tensor) -> Tensor:
        """
        theta: [..., dim_theta]
        Returns: [...]
        """
        if self.family == "normal":
            
            return self.base.log_prob(theta - self.misspec_shift)

        # lognormal: theta must be positive
        theta_pos = theta.clamp_min(self.eps)
        z = torch.log(theta_pos)

        lp = self.base.log_prob(z)

        lp = lp - torch.sum(torch.log(theta_pos), dim=-1)
        return lp


class SimplePendulum(Task):

    def __init__(
        self,
        g: float = 9.81,
        d: float = 0.1,
        dt: float = 0.05,
        dim_theta: int = 3,
        dim_xi: int = 1, 
        dim_y: int = 2,
        prior: Optional[PriorPendulum] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        _seed: Optional[int] = None,
    ) -> None:
        
        if prior is not None:
            prior = prior
        else:
            prior = PriorPendulum(dim_theta=dim_theta, device=device, dtype=dtype)

        super().__init__(prior=prior, device=device)

        self.dtype = dtype
        self.g = g
        self.d = d
        self.dt = dt
        self.dim_theta = dim_theta
        self.dim_xi = dim_xi
        self.dim_y = dim_y

        self.generator = torch.Generator(device=self.device)
        if _seed is not None:
            self.generator.manual_seed(_seed)

        self.x0 = self.get_initial_observation()
    

    def __repr__(self):
        return NotImplementedError("Implement a nice string representation of the task here.")
    

    def get_initial_observation(self, n_samples: int = 1) -> Tensor:
        """
        Returns: [dim_y, ] initial observation of the system (initial angulars position and velocity)
        """
        return torch.zeros((n_samples, self.dim_y), device=self.device, dtype=self.dtype)
    
    
    def compute_sde_coeffs_old(self, theta: Tensor, y: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """
        theta: [batch, dim_theta] 
        xi: [batch, dim_xi]
        y: [batch, dim_y]
        Returns:
            f: [batch, dim_y] drift coefficients
            g: [batch, dim_y] diffusion coefficients
        """
        assert theta.shape[-1] == self.dim_theta, f"Expected theta with last dim {self.dim_theta}, but got {theta.shape[-1]}"
        assert y.shape[-1] == self.dim_y, f"Expected y with last dim {self.dim_y}, but got {y.shape[-1]}"
        assert xi.shape[-1] == self.dim_xi, f"Expected xi with last dim {self.dim_xi}, but got {xi.shape[-1]}"
        assert theta.device == y.device == xi.device == self.device, f"Expected theta, y, and xi to be on device {self.device}, but got theta on {theta.device}, y on {y.device}, and xi on {xi.device}"

        B = theta.shape[0]
        dx1 = y[:, 1:2]

        h1 = -torch.sin(y[:, 0:1]) * theta[:, 0:1]
        h2 = -y[:, 1:2] * theta[:, 1:2]
        h3 = xi * theta[:, 2:3]
        dx2 = h1 + h2 + h3

        drift = torch.cat([dx1, dx2], dim=-1) ###  pas besoin de mettre sur device puisque les tenseurs d'entrée sont déjà sur le bon device et que les opérations préservent le device    
        diffusion = torch.tensor([0, 0.1], device=self.device, dtype=self.dtype).expand(B, self.dim_y) ## On pourrait enregister un buffer pour eviter de le creer a chaque fois mais osef

        return drift, diffusion
    
    def compute_sde_coeffs(self, theta: Tensor, y: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """
        theta: [..., dim_theta]
        y:     [..., dim_y]
        xi:    [..., dim_xi]

        Returns:
            drift:     [..., dim_y]
            diffusion: [..., dim_y]
        """
        if theta.shape[-1] != self.dim_theta:
            raise ValueError(
                f"Expected theta.shape[-1] == {self.dim_theta}, got {theta.shape}"
            )
        if y.shape[-1] != self.dim_y:
            raise ValueError(
                f"Expected y.shape[-1] == {self.dim_y}, got {y.shape}"
            )
        if xi.shape[-1] != self.dim_xi:
            raise ValueError(
                f"Expected xi.shape[-1] == {self.dim_xi}, got {xi.shape}"
            )

        if not (theta.device == y.device == xi.device):
            raise ValueError(
                f"Expected theta, y, xi on same device, got "
                f"theta={theta.device}, y={y.device}, xi={xi.device}"
            )

        # Extract components
        x1 = y[..., 0:1]
        x2 = y[..., 1:2]

        theta1 = theta[..., 0:1]
        theta2 = theta[..., 1:2]
        theta3 = theta[..., 2:3]

        # Broadcast everything to a common leading shape
        x1, x2, xi, theta1, theta2, theta3 = torch.broadcast_tensors(
            x1, x2, xi, theta1, theta2, theta3
        )

        dx1 = x2
        h1 = -torch.sin(x1) * theta1
        h2 = -x2 * theta2
        h3 = xi * theta3
        dx2 = h1 + h2 + h3

        drift = torch.cat([dx1, dx2], dim=-1)

        diffusion_base = torch.tensor(
            [0.0, 0.1],
            device=y.device,
            dtype=y.dtype,
        )
        diffusion = torch.broadcast_to(diffusion_base, drift.shape)

        return drift, diffusion
    
    def _sample_outcome(self, theta: Tensor, design: Tensor, last_y: Tensor, angular_constraint: bool = True)-> Tensor:
        """
        theta: [batch, dim_theta]
        design: [batch, dim_xi]
        last_y: [batch, dim_y]
        Returns:
            y_next: [batch, dim_y] next observation after applying design for one time step
        """
        B, D = last_y.shape
        assert D == self.dim_y, f"Expected last_y with last dim {self.dim_y}, but got {D}"
        assert design.shape == (B, self.dim_xi), f"Expected design with shape {(B, self.dim_xi)}, but got {design.shape}"
        assert theta.shape == (B, self.dim_theta), f"Expected theta with shape {(B, self.dim_theta)}, but got {theta.shape}"

        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)
        noise = torch.randn(drift.shape, device=drift.device, dtype=drift.dtype, generator=self.generator) * math.sqrt(self.dt)
        y_next = last_y + drift * self.dt + diffusion * noise
        
        if angular_constraint:
            y_next[:, 0] = (y_next[:, 0] + math.pi) % (2 * math.pi) - math.pi  # wrap angle to [-pi, pi]
        return y_next

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        last_y: Tensor,
    ) -> Tensor:
        """
        Generic log p(y | theta, design, last_y) with support for extra leading dims.

        Accepted examples:
            y, design, last_y: [B, Dy], [B, Dxi], [B, Dy]
            theta:            [B, Dtheta]
            -> returns [B]

            y, design, last_y: [B, T, Dy], [B, T, Dxi], [B, T, Dy]
            theta:             [B, Dtheta]
            -> returns [B, T]

            y, design, last_y: [B, L, T, Dy], [B, L, T, Dxi], [B, L, T, Dy]
            theta:             [B, L, T, Dtheta] or [B, L, Dtheta]
            -> returns [B, L, T]

        Convention:
            only the second state dimension is observed with noise, so the returned
            log-probability corresponds to that marginal observation model.
        """
        if y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected y.shape[-1] == {self.dim_y}, got {y.shape}")
        if last_y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected last_y.shape[-1] == {self.dim_y}, got {last_y.shape}")
        if design.shape[-1] != self.dim_xi:
            raise ValueError(f"Expected design.shape[-1] == {self.dim_xi}, got {design.shape}")
        if theta.shape[-1] != self.dim_theta:
            raise ValueError(f"Expected theta.shape[-1] == {self.dim_theta}, got {theta.shape}")

        # compute_sde_coeffs should support broadcasting on leading dimensions
        # and return tensors of shape [..., dim_y]
        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)

        if drift.shape[-1] != self.dim_y:
            raise ValueError(f"Expected drift.shape[-1] == {self.dim_y}, got {drift.shape}")
        if diffusion.shape[-1] != self.dim_y:
            raise ValueError(f"Expected diffusion.shape[-1] == {self.dim_y}, got {diffusion.shape}")

        dist = torch_dist.Normal(
            loc=last_y[..., 1] + drift[..., 1] * self.dt,
            scale=diffusion[..., 1] * math.sqrt(self.dt),
        )

        logp = dist.log_prob(y[..., 1])  # shape: leading dims of y/theta/design/last_y
        return logp
    

    def _outcome_logp_contrastive(
        self,
        y: Tensor,         # [B, T, dim_y]
        theta: Tensor,     # [B, L, dim_theta]
        design: Tensor,    # [B, T, dim_xi]
        last_y: Tensor,    # [B, T, dim_y]
    ) -> Tensor:
        """
        Compute log p(y_t | theta_l, design_t, last_y_t) for all contrastive thetas.

        Args:
            y:       [B, T, dim_y]
            theta:   [B, L, dim_theta]
            design:  [B, T, dim_xi]
            last_y:  [B, T, dim_y]

        Returns:
            logp:    [B, L, T]
        """
        if y.ndim != 3 or y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected y of shape [B, T, {self.dim_y}], got {y.shape}")
        if last_y.ndim != 3 or last_y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected last_y of shape [B, T, {self.dim_y}], got {last_y.shape}")
        if design.ndim != 3 or design.shape[-1] != self.dim_xi:
            raise ValueError(f"Expected design of shape [B, T, {self.dim_xi}], got {design.shape}")
        if theta.ndim != 3 or theta.shape[-1] != self.dim_theta:
            raise ValueError(f"Expected theta of shape [B, L, {self.dim_theta}], got {theta.shape}")

        B, T, _ = y.shape
        B_theta, L, _ = theta.shape
        B_design, T_design, _ = design.shape
        B_last, T_last, _ = last_y.shape

        if B_theta != B or B_design != B or B_last != B:
            raise ValueError(
                f"Incompatible batch sizes: y={y.shape}, theta={theta.shape}, "
                f"design={design.shape}, last_y={last_y.shape}"
            )
        if T_design != T or T_last != T:
            raise ValueError(
                f"Incompatible time sizes: y={y.shape}, design={design.shape}, last_y={last_y.shape}"
            )

        # Expand to [B, L, T, ...]
        y_exp = y[:, None, :, :]          # [B, 1, T, dim_y]
        design_exp = design[:, None, :, :]  # [B, 1, T, dim_xi]
        last_y_exp = last_y[:, None, :, :]  # [B, 1, T, dim_y]
        theta_exp = theta[:, :, None, :]    # [B, L, 1, dim_theta]

        logp = self._outcome_logp(
            y=y_exp,
            theta=theta_exp,
            design=design_exp,
            last_y=last_y_exp,
        )  # [B, L, T]

        return logp
    
    # --------------------- Reparametrized Version of Sampling Method   ------------------------------------------
    def reparam_sample_residuals(self, B: int, T: int) -> Tensor:
        """
        B: number of samples to draw
        Returns:
            eps: [B, dim_y] standard normal residuals for reparameterized sampling
        """
        eps = torch.randn((B, T, self.dim_y), device=self.device, dtype=self.dtype, generator=self.generator)

        return eps * math.sqrt(self.dt)
    
    def reparam_make_outcome(
        self, 
        theta: Tensor, 
        design: Tensor, 
        last_y: Tensor, 
        eps: Tensor,
        angular_constraint: bool = True
    ) -> Tensor:
        """
        theta: [batch, dim_theta]
        design: [batch, dim_xi]
        last_y: [batch, dim_y]
        eps: [batch, dim_y] pre-sampled residuals for reparameterized sampling
        Returns:
            y_next: [batch, dim_y] next observation after applying design for one time step
        """
        B, D = last_y.shape

        assert D == self.dim_y, f"Expected last_y with last dim {self.dim_y}, but got {D}"
        assert design.shape == (B, self.dim_xi), f"Expected design with shape {(B, self.dim_xi)}, but got {design.shape}"
        assert theta.shape == (B, self.dim_theta), f"Expected theta with shape {(B, self.dim_theta)}, but got {theta.shape}"

        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)
        y_next = last_y + drift * self.dt + diffusion * eps                 # Multiplication par dt dans le noise déjà faite 
        if angular_constraint:
            y_next[:, 0] = (y_next[:, 0] + math.pi) % (2 * math.pi) - math.pi  # wrap angle to [-pi, pi]
        return y_next


class DoublePendulum(Task):

    def __init__(
        self,
        g: float = 9.81,
        d: float = 0.1,
        dt: float = 0.05,
        dim_theta: int = 4,
        dim_xi: int = 2, 
        dim_y: int = 4,
        prior: Optional[PriorPendulum] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
        _seed: Optional[int] = None,
    ) -> None:
        
        if prior is not None:
            prior = prior
        else:
            mu_log = torch.zeros(dim_theta, device=device, dtype=dtype)         # mean of log(theta)
            cov_log = (0.1**2) * torch.ones(dim_theta, device=device, dtype=dtype)

            prior = PriorPendulum(
                dim_theta=dim_theta,
                mu_theta=mu_log,
                cov_theta=cov_log,
                device=device,
                dtype=dtype,
                family="lognormal",
            )


        super().__init__(device=device, prior=prior)

        self.g = g
        self.d = d
        self.dt = dt
        self.dtype = dtype
        self.dim_theta = dim_theta # (m1, l1, m2, l2)
        self.dim_xi = dim_xi     # (torque on pendulum 1, torque on pendulum 2)
        self.dim_y = dim_y      # (theta1, theta2, theta1_dot, theta2_dot)

        self.generator = torch.Generator(device=self.device)
        if _seed is not None:
            self.generator.manual_seed(_seed)

        self.x0 = self.get_initial_observation()

    
    def __repr__(self):
        return NotImplementedError("Implement a nice string representation of the task here.")
    
    
    def get_initial_observation(self, n_samples: int = 1) -> Tensor:
        """
        Returns: [dim_y, ] initial observation of the system (initial angulars position and velocity)
        """
        return torch.zeros((n_samples, self.dim_y), device=self.device, dtype=self.dtype)
    

    def compute_sde_coeffs(self, theta: Tensor, y: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """
        theta: [..., 4] with (m1, m2, l1, l2)
        y:     [..., 4] with (q1, q2, q1dot, q2dot)
        xi:    [..., 2] with (xi1, xi2)

        Returns:
            drift:     [..., 4]
            diffusion: [..., 4]
        """
        if theta.shape[-1] != self.dim_theta:
            raise ValueError(f"Expected theta.shape[-1] == {self.dim_theta}, got {theta.shape}")
        if y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected y.shape[-1] == {self.dim_y}, got {y.shape}")
        if xi.shape[-1] != self.dim_xi:
            raise ValueError(f"Expected xi.shape[-1] == {self.dim_xi}, got {xi.shape}")

        if not (theta.device == y.device == xi.device):
            raise ValueError(
                f"Expected theta, y, xi on same device, got "
                f"theta={theta.device}, y={y.device}, xi={xi.device}"
            )

        dtype = y.dtype
        device = y.device

        q1  = y[..., 0:1]
        q2  = y[..., 1:2]
        q1d = y[..., 2:3]
        q2d = y[..., 3:4]

        m1 = theta[..., 0:1]
        m2 = theta[..., 1:2]
        l1 = theta[..., 2:3]
        l2 = theta[..., 3:4]

        xi1 = xi[..., 0:1]
        xi2 = xi[..., 1:2]

        q1, q2, q1d, q2d, m1, m2, l1, l2, xi1, xi2 = torch.broadcast_tensors(
            q1, q2, q1d, q2d, m1, m2, l1, l2, xi1, xi2
        )

        g_const = getattr(self, "g", 9.81)

        cos_q2 = torch.cos(q2)
        sin_q2 = torch.sin(q2)

        M11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2.0 * m2 * l1 * l2 * cos_q2
        M12 = m2 * l2 * l2 + m2 * l1 * l2 * cos_q2
        M22 = m2 * l2 * l2

        common = m2 * l1 * l2 * sin_q2
        two_q1d_plus_q2d = 2.0 * q1d + q2d

        C12 = -common * two_q1d_plus_q2d
        C21 = 0.5 * common * two_q1d_plus_q2d
        C22 = -0.5 * common * q1d

        Cqdot_1 = C12 * q2d
        Cqdot_2 = C21 * q1d + C22 * q2d

        sin_q1 = torch.sin(q1)
        sin_q1q2 = torch.sin(q1 + q2)

        tau1 = -g_const * ((m1 + m2) * l1 * sin_q1 + m2 * l2 * sin_q1q2)
        tau2 = -g_const * (m2 * l2 * sin_q1q2)

        rhs1 = tau1 + xi1 - Cqdot_1
        rhs2 = tau2 + xi2 - Cqdot_2

        M = torch.stack(
            [
                torch.cat([M11, M12], dim=-1),
                torch.cat([M12, M22], dim=-1),
            ],
            dim=-2,
        )  # [..., 2, 2]

        rhs = torch.cat([rhs1, rhs2], dim=-1)  # [..., 2]

        eye = torch.eye(2, device=device, dtype=dtype)
        eye = eye.expand(M.shape[:-2] + (2, 2))

        lam = 1e-6
        qdd = torch.linalg.solve(M + lam * eye, rhs.unsqueeze(-1)).squeeze(-1)

        q1dd = qdd[..., 0:1]
        q2dd = qdd[..., 1:2]

        drift = torch.cat([q1d, q2d, q1dd, q2dd], dim=-1)

        diffusion_base = torch.tensor(
            [0.0, 0.0, 0.1, 0.1],
            device=device,
            dtype=dtype,
        )
        diffusion = torch.broadcast_to(diffusion_base, drift.shape)

        return drift, diffusion


    def _sample_outcome(self, theta: Tensor, design: Tensor, last_y: Tensor, angular_constraint: bool = True) -> Tensor:
        """
        Sample next state given current state, parameters, and design.

        theta: [B, 4]  with (m1, m2, l1, l2)
        last_y: [B, 4] with (q1, q2, q1dot, q2dot)
        design: [B, 2] with (xi1, xi2)

        Returns
        -------
        y_next : [B, 4] with (q1_next, q2_next, q1dot_next, q2dot_next)
        """
        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)
        noise = torch.randn(drift.shape, device=drift.device, dtype=drift.dtype, generator=self.generator) * math.sqrt(self.dt)
        y_next = last_y + drift * self.dt + diffusion * noise

        if angular_constraint:
            y_next[:, 0] = (y_next[:, 0] + math.pi) % (2 * math.pi) - math.pi  # wrap q1 to [-pi, pi]
            y_next[:, 1] = (y_next[:, 1] + math.pi) % (2 * math.pi) - math.pi  # wrap q2 to [-pi, pi]

        return y_next
    

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        last_y: Tensor,
    ) -> Tensor:
        """
        Generic log p(y | theta, design, last_y), compatible with sPCE broadcasting.

        Examples:
            y/design/last_y: [B,T,*], theta: [B,Dtheta]       -> [B,T]
            y/design/last_y: [B,L,T,*], theta: [B,L,1,Dtheta] -> [B,L,T]

        Only velocity dimensions 2 and 3 are stochastic.
        """
        if y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected y.shape[-1] == {self.dim_y}, got {y.shape}")
        if last_y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected last_y.shape[-1] == {self.dim_y}, got {last_y.shape}")
        if design.shape[-1] != self.dim_xi:
            raise ValueError(f"Expected design.shape[-1] == {self.dim_xi}, got {design.shape}")
        if theta.shape[-1] != self.dim_theta:
            raise ValueError(f"Expected theta.shape[-1] == {self.dim_theta}, got {theta.shape}")

        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)

        loc_vel = last_y[..., 2:4] + drift[..., 2:4] * self.dt
        scale_vel = diffusion[..., 2:4] * math.sqrt(self.dt)

        dist = torch_dist.Normal(loc=loc_vel, scale=scale_vel)

        logp = dist.log_prob(y[..., 2:4]).sum(dim=-1)

        return logp
    
    def _outcome_logp_contrastive(
        self,
        y: Tensor,         # [B, T, dim_y]
        theta: Tensor,     # [B, L, dim_theta]
        design: Tensor,    # [B, T, dim_xi]
        last_y: Tensor,    # [B, T, dim_y]
    ) -> Tensor:
        """
        Compute log p(y_t | theta_l, design_t, last_y_t) for all contrastive thetas.

        Returns:
            logp: [B, L, T]
        """
        if y.ndim != 3 or y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected y of shape [B, T, {self.dim_y}], got {y.shape}")
        if last_y.ndim != 3 or last_y.shape[-1] != self.dim_y:
            raise ValueError(f"Expected last_y of shape [B, T, {self.dim_y}], got {last_y.shape}")
        if design.ndim != 3 or design.shape[-1] != self.dim_xi:
            raise ValueError(f"Expected design of shape [B, T, {self.dim_xi}], got {design.shape}")
        if theta.ndim != 3 or theta.shape[-1] != self.dim_theta:
            raise ValueError(f"Expected theta of shape [B, L, {self.dim_theta}], got {theta.shape}")

        B, T, _ = y.shape
        B_theta, L, _ = theta.shape

        if B_theta != B or design.shape[0] != B or last_y.shape[0] != B:
            raise ValueError(
                f"Incompatible batch sizes: y={y.shape}, theta={theta.shape}, "
                f"design={design.shape}, last_y={last_y.shape}"
            )
        if design.shape[1] != T or last_y.shape[1] != T:
            raise ValueError(
                f"Incompatible time sizes: y={y.shape}, design={design.shape}, last_y={last_y.shape}"
            )

        y_exp = y[:, None, :, :]             # [B, 1, T, dim_y]
        design_exp = design[:, None, :, :]   # [B, 1, T, dim_xi]
        last_y_exp = last_y[:, None, :, :]   # [B, 1, T, dim_y]
        theta_exp = theta[:, :, None, :]     # [B, L, 1, dim_theta]

        return self._outcome_logp(
            y=y_exp,
            theta=theta_exp,
            design=design_exp,
            last_y=last_y_exp,
        )  # [B, L, T]
    
    # --------------------- Reparametrized Version of Sampling Method   ------------------------------------------
    def reparam_sample_residuals(self, B: int, T: int) -> Tensor:
        """
        Sample standard normal residuals for reparameterized sampling.

        B: number of samples
        T: time horizon (number of steps)

        Returns
        -------
        eps : [B, T, 4] standard normal residuals for reparameterized sampling
        """
        eps = torch.randn((B, T, self.dim_y), device=self.device, dtype=self.dtype, generator=self.generator)
        return eps * math.sqrt(self.dt)
    
    def reparam_make_outcome(
        self, 
        theta: Tensor, 
        design: Tensor, 
        last_y: Tensor, 
        eps: Tensor,
        angular_constraint: bool = True
    ) -> Tensor:
        """
        Create next state using reparameterized sampling.

        theta: [B, 4] with (m1, m2, l1, l2)
        design: [B, 2] with (xi1, xi2)
        last_y: [B, 4] with (q1, q2, q1dot, q2dot)
        eps: [B, 4] standard normal residuals for reparameterized sampling

        Returns
        -------
        y_next : [B, 4] with (q1_next, q2_next, q1dot_next, q2dot_next)
        """
        drift, diffusion = self.compute_sde_coeffs(theta, last_y, design)
        y_next = last_y + drift * self.dt + diffusion * eps

        if angular_constraint:
            y_next[:, 0] = (y_next[:, 0] + math.pi) % (2 * math.pi) - math.pi  # wrap q1 to [-pi, pi]
            y_next[:, 1] = (y_next[:, 1] + math.pi) % (2 * math.pi) - math.pi  # wrap q2 to [-pi, pi]

        return y_next
