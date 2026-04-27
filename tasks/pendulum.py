import torch
from attrdictionary import AttrDict

from simulators import SimplePendulum, PriorPendulum, DoublePendulum


class HiddenPendulum:
    def __init__(
        self,
        dim_x=1,
        dim_y=2,
        dim_theta=3,
        n_target_theta=3,
        n_context_init=1,
        n_query_init=200,
        design_scale=1.0,
        outcome_scale=1.0,
        dt=0.05,
        theta_family="normal",
        device="cpu",
        training=True,
        **kwargs,
    ):
        self.training = training
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.device = torch.device(device)

        prior = PriorPendulum(
            dim_theta=dim_theta,
            family=theta_family,
            device=self.device,
        )

        self.simulator = SimplePendulum(
            dt=dt,
            dim_theta=dim_theta,
            dim_xi=dim_x,
            dim_y=dim_y,
            prior=prior,
            device=self.device,
        )

    def __repr__(self):
        return (
            f"HiddenPendulum(training={self.training}, "
            f"dim_x={self.dim_x}, dim_y={self.dim_y}, "
            f"dim_theta={self.dim_theta}, device={self.device})"
        )

    def sample_theta(self, batch_size):
        return self.simulator.prior.sample(batch_size)

    def sample_query_x(self, batch_size):
        # Candidate controls xi in [-design_scale, design_scale]
        return (
            2.0 * self.design_scale
            * torch.rand(batch_size, self.n_query_init, self.dim_x, device=self.device)
            - self.design_scale
        )

    def sample_batch(self, batch_size):
        theta = self.sample_theta(batch_size)  # [B, dim_theta]

        query_x = self.sample_query_x(batch_size)  # [B, n_query, dim_x]

        # Initial physical state y0 = (angle, angular velocity)
        current_y = self.simulator.get_initial_observation(batch_size)  # [B, dim_y]

        # ALINE expects at least one context point if n_context_init=1.
        # We create a dummy initial context with xi=0 and y=y0.
        context_x = torch.zeros(
            batch_size,
            self.n_context_init,
            self.dim_x,
            device=self.device,
        )

        context_y = current_y[:, None, :].repeat(1, self.n_context_init, 1)

        batch = AttrDict()
        batch.context_x = context_x
        batch.context_y = context_y
        batch.query_x = query_x
        batch.target_all = theta
        batch.theta = theta
        batch.current_y = current_y

        return batch

    def update_batch(self, batch, idx):
        """
        idx: [B] or [B, 1], selected candidate indices from batch.query_x.
        """
        if idx.ndim > 1:
            idx = idx.squeeze(-1)

        B = batch.query_x.shape[0]
        arange = torch.arange(B, device=self.device)

        xi = batch.query_x[arange, idx]  # [B, dim_x]

        y_next = self.simulator._sample_outcome(
            theta=batch.theta,
            design=xi,
            last_y=batch.current_y,
        )  # [B, dim_y]

        batch.context_x = torch.cat(
            [batch.context_x, xi[:, None, :]],
            dim=1,
        )

        batch.context_y = torch.cat(
            [batch.context_y, y_next[:, None, :]],
            dim=1,
        )

        batch.current_y = y_next

        # Resample candidates after each step.
        batch.query_x = self.sample_query_x(B)

        return batch


class HiddenDoublePendulum:
    def __init__(
        self,
        dim_x=2,
        dim_y=4,
        dim_theta=4,
        n_target_theta=4,
        n_context_init=1,
        n_query_init=200,
        design_scale=1.0,
        outcome_scale=1.0,
        dt=0.05,
        device="cpu",
        training=True,
        **kwargs,
    ):
        self.training = training
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.device = torch.device(device)

        self.simulator = DoublePendulum(
            dt=dt,
            dim_theta=dim_theta,
            dim_xi=dim_x,
            dim_y=dim_y,
            device=self.device,
        )

    def __repr__(self):
        return (
            f"HiddenDoublePendulum(training={self.training}, "
            f"dim_x={self.dim_x}, dim_y={self.dim_y}, "
            f"dim_theta={self.dim_theta}, device={self.device})"
        )

    def sample_theta(self, batch_size):
        return self.simulator.prior.sample(batch_size)

    def sample_query_x(self, batch_size):
        return (
            2.0 * self.design_scale
            * torch.rand(batch_size, self.n_query_init, self.dim_x, device=self.device)
            - self.design_scale
        )

    def sample_batch(self, batch_size):
        theta = self.sample_theta(batch_size)
        query_x = self.sample_query_x(batch_size)

        current_y = self.simulator.get_initial_observation(batch_size)

        context_x = torch.zeros(
            batch_size,
            self.n_context_init,
            self.dim_x,
            device=self.device,
        )

        context_y = current_y[:, None, :].repeat(1, self.n_context_init, 1)

        batch = AttrDict()
        batch.context_x = context_x
        batch.context_y = context_y
        batch.query_x = query_x

        # Target predicted by ALINE
        batch.target_all = theta
        batch.theta = theta

        # Stateful dynamics
        batch.current_y = current_y

        return batch

    def update_batch(self, batch, idx):
        if idx.ndim > 1:
            idx = idx.squeeze(-1)

        B = batch.query_x.shape[0]
        arange = torch.arange(B, device=self.device)

        xi = batch.query_x[arange, idx]  # [B, 2]

        y_next = self.simulator._sample_outcome(
            theta=batch.theta,
            design=xi,
            last_y=batch.current_y,
        )

        batch.context_x = torch.cat(
            [batch.context_x, xi[:, None, :]],
            dim=1,
        )

        batch.context_y = torch.cat(
            [batch.context_y, y_next[:, None, :]],
            dim=1,
        )

        batch.current_y = y_next
        batch.query_x = self.sample_query_x(B)

        return batch


