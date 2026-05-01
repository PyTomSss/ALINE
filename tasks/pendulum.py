import torch
from attrdictionary import AttrDict
from tasks.base_task import Task
from tasks.simulators import SimplePendulum, PriorPendulum, DoublePendulum


class HiddenPendulum_old(Task):
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
        super(HiddenPendulum, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta)
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
        batch.target_theta = theta
        batch.n_target_theta = self.n_target_theta

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


class HiddenPendulum(Task):
    """
    ALINE wrapper for a hidden-parameter stochastic pendulum task.

    The true physical simulator uses controls xi_t of dimension dim_xi.

    ALINE, however, receives augmented transition inputs:

        x_t = [xi_t, y_{t-1}, time_t]

    and outputs/observes either:

        y_context_t = y_t - y_{t-1}

    or, if use_delta_y=False:

        y_context_t = y_t

    This is important because the pendulum likelihood is sequential:

        p(y_t | theta, xi_t, y_{t-1})

    rather than exchangeable:

        p(y_t | theta, xi_t).
    """

    def __init__(
        self,
        dim_xi=1,
        dim_y=2,
        dim_theta=3,
        n_target_theta=3,
        n_context_init=1,
        n_query_init=200,
        design_scale=1.0,
        outcome_scale=1.0,
        dt=0.05,
        theta_family="normal",
        include_time=True,
        normalize_time=True,
        max_T=5,
        use_delta_y=True,
        device="cpu",
        training=True,
        **kwargs,
    ):
        self.dim_xi = dim_xi
        self.dim_y = dim_y
        self.dim_theta = dim_theta
        self.n_target_theta = n_target_theta

        self.n_context_init = n_context_init
        self.n_query_init = n_query_init

        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.dt = dt

        self.include_time = include_time
        self.normalize_time = normalize_time
        self.max_T = max_T
        self.use_delta_y = use_delta_y

        self.training = training
        self.device = torch.device(device)

        dim_x_aline = dim_xi + dim_y + int(include_time)

        super(HiddenPendulum, self).__init__(
            dim_x=dim_x_aline,
            dim_y=dim_y,
            n_target_theta=n_target_theta,
        )

        self.dim_x = dim_x_aline

        prior = PriorPendulum(
            dim_theta=dim_theta,
            family=theta_family,
            device=self.device,
        )

        self.simulator = SimplePendulum(
            dt=dt,
            dim_theta=dim_theta,
            dim_xi=dim_xi,
            dim_y=dim_y,
            prior=prior,
            device=self.device,
        )

    def __repr__(self):
        return (
            f"HiddenPendulum(training={self.training}, "
            f"dim_xi={self.dim_xi}, dim_x={self.dim_x}, dim_y={self.dim_y}, "
            f"dim_theta={self.dim_theta}, include_time={self.include_time}, "
            f"use_delta_y={self.use_delta_y}, device={self.device})"
        )

    def sample_theta(self, batch_size):
        return self.simulator.prior.sample(batch_size)

    def _time_feature(self, batch_size, step, dtype=torch.float32):
        """
        Returns a time feature of shape [B, 1].

        If normalize_time=True:
            time = step / max_T
        otherwise:
            time = step
        """
        if self.normalize_time:
            denom = max(float(self.max_T), 1.0)
            value = float(step) / denom
        else:
            value = float(step)

        return torch.full(
            (batch_size, 1),
            value,
            device=self.device,
            dtype=dtype,
        )

    def make_aline_x(self, xi, prev_y, step):
        """
        Build the augmented ALINE input.

        Parameters
        ----------
        xi:
            Tensor of shape [B, dim_xi].
        prev_y:
            Tensor of shape [B, dim_y].
        step:
            Integer time step.

        Returns
        -------
        x:
            Tensor of shape [B, dim_x], where

                x = [xi, prev_y, time_feature]

            if include_time=True, otherwise

                x = [xi, prev_y].
        """
        if xi.ndim != 2:
            raise ValueError(f"xi must have shape [B, dim_xi], got {xi.shape}")

        if prev_y.ndim != 2:
            raise ValueError(f"prev_y must have shape [B, dim_y], got {prev_y.shape}")

        if xi.shape[0] != prev_y.shape[0]:
            raise ValueError(
                f"xi and prev_y must have same batch size, got "
                f"{xi.shape[0]} and {prev_y.shape[0]}"
            )

        if xi.shape[-1] != self.dim_xi:
            raise ValueError(
                f"xi last dim should be dim_xi={self.dim_xi}, got {xi.shape[-1]}"
            )

        if prev_y.shape[-1] != self.dim_y:
            raise ValueError(
                f"prev_y last dim should be dim_y={self.dim_y}, got {prev_y.shape[-1]}"
            )

        pieces = [xi, prev_y]

        if self.include_time:
            time_feat = self._time_feature(
                batch_size=xi.shape[0],
                step=step,
                dtype=xi.dtype,
            )
            pieces.append(time_feat)

        x = torch.cat(pieces, dim=-1)

        if x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"Internal error: augmented x has dim {x.shape[-1]}, "
                f"expected {self.dim_x}"
            )

        return x

    def sample_query_x(self, batch_size, current_y=None, step=0):
        """
        Sample candidate controls and augment them with the current state
        and optional time feature.

        Returns
        -------
        query_x:
            Tensor of shape [B, n_query_init, dim_x].

            Each candidate has the form:

                [xi_candidate, current_y, time_feature].
        """
        xi_candidates = (
            2.0
            * self.design_scale
            * torch.rand(
                batch_size,
                self.n_query_init,
                self.dim_xi,
                device=self.device,
            )
            - self.design_scale
        )

        if current_y is None:
            current_y = torch.zeros(
                batch_size,
                self.dim_y,
                device=self.device,
                dtype=xi_candidates.dtype,
            )

        if current_y.ndim != 2:
            raise ValueError(
                f"current_y must have shape [B, dim_y], got {current_y.shape}"
            )

        if current_y.shape[0] != batch_size:
            raise ValueError(
                f"current_y batch size should be {batch_size}, "
                f"got {current_y.shape[0]}"
            )

        if current_y.shape[-1] != self.dim_y:
            raise ValueError(
                f"current_y last dim should be dim_y={self.dim_y}, "
                f"got {current_y.shape[-1]}"
            )

        prev_y_rep = current_y[:, None, :].expand(
            batch_size,
            self.n_query_init,
            self.dim_y,
        )

        pieces = [xi_candidates, prev_y_rep]

        if self.include_time:
            time_feat = self._time_feature(
                batch_size=batch_size,
                step=step,
                dtype=xi_candidates.dtype,
            )
            time_feat = time_feat[:, None, :].expand(
                batch_size,
                self.n_query_init,
                1,
            )
            pieces.append(time_feat)

        query_x = torch.cat(pieces, dim=-1)

        if query_x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"query_x has last dim {query_x.shape[-1]}, expected {self.dim_x}"
            )

        return query_x

    def sample_batch(self, batch_size):
        """
        Sample a fresh ALINE batch.

        The initial context is a dummy transition starting from y0 with xi=0.
        If use_delta_y=True, its context_y is zero.
        """
        theta = self.sample_theta(batch_size)  # [B, dim_theta]

        current_y = self.simulator.get_initial_observation(batch_size)  # [B, dim_y]
        current_y = current_y.to(self.device)

        step = 0

        query_x = self.sample_query_x(
            batch_size=batch_size,
            current_y=current_y,
            step=step,
        )

        xi0 = torch.zeros(
            batch_size,
            self.dim_xi,
            device=self.device,
            dtype=current_y.dtype,
        )

        context_x0 = self.make_aline_x(
            xi=xi0,
            prev_y=current_y,
            step=step,
        )  # [B, dim_x]

        context_x = context_x0[:, None, :].repeat(
            1,
            self.n_context_init,
            1,
        )

        if self.use_delta_y:
            context_y0 = torch.zeros_like(current_y)
        else:
            context_y0 = current_y

        context_y = context_y0[:, None, :].repeat(
            1,
            self.n_context_init,
            1,
        )

        batch = AttrDict()
        batch.context_x = context_x
        batch.context_y = context_y
        batch.query_x = query_x

        batch.target_all = theta
        batch.theta = theta
        batch.target_theta = theta
        batch.n_target_theta = self.n_target_theta

        batch.current_y = current_y
        batch.t = step

        return batch

    def update_batch(self, batch, idx):
        """
        Update the batch after ALINE selects one candidate query.

        Parameters
        ----------
        batch:
            AttrDict containing context_x, context_y, query_x, theta, current_y.
        idx:
            Tensor of shape [B] or [B, 1], containing the selected candidate indices.

        Notes
        -----
        batch.query_x contains augmented candidates:

            [xi_candidate, current_y, time_feature]

        but the simulator only receives the physical control xi_candidate.
        """
        if idx.ndim > 1:
            idx = idx.squeeze(-1)

        B = batch.query_x.shape[0]

        if idx.shape[0] != B:
            raise ValueError(
                f"idx batch size should be {B}, got {idx.shape[0]}"
            )

        arange = torch.arange(B, device=self.device)

        selected_x = batch.query_x[arange, idx]  # [B, dim_x]

        if selected_x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"selected_x has dim {selected_x.shape[-1]}, expected {self.dim_x}"
            )

        #3xtract the true physical control
        xi = selected_x[:, : self.dim_xi]  # [B, dim_xi]

        prev_y = batch.current_y  # [B, dim_y]

        y_next = self.simulator._sample_outcome(
            theta=batch.theta,
            design=xi,
            last_y=prev_y,
        )  # [B, dim_y]

        if self.outcome_scale != 1.0:
        
            observed_delta = (y_next - prev_y) / self.outcome_scale
            observed_y = y_next / self.outcome_scale
        else:
            observed_delta = y_next - prev_y
            observed_y = y_next

        if self.use_delta_y:
            context_y_new = observed_delta
        else:
            context_y_new = observed_y

        batch.context_x = torch.cat(
            [batch.context_x, selected_x[:, None, :]],
            dim=1,
        )

        batch.context_y = torch.cat(
            [batch.context_y, context_y_new[:, None, :]],
            dim=1,
        )

        batch.current_y = y_next

        batch.t = int(batch.t) + 1

        # Resample candidate physical controls and augment with the new state
        batch.query_x = self.sample_query_x(
            batch_size=B,
            current_y=batch.current_y,
            step=batch.t,
        )

        return batch




# ------------------ Double Pendulum Wrapper for ALINE ------------- #
class HiddenDoublePendulum:
    """
    ALINE wrapper for the hidden-parameter stochastic double pendulum.

    Physical simulator:
        xi_t in R^2
        y_t  in R^4
        theta in R^4

    ALINE input is augmented as:

        x_t = [xi_t, y_{t-1}, time_t]

    if include_time=True, otherwise:

        x_t = [xi_t, y_{t-1}]

    The ALINE observation is either:

        context_y_t = y_t - y_{t-1}

    if use_delta_y=True, or:

        context_y_t = y_t

    if use_delta_y=False.
    """

    def __init__(
        self,
        dim_xi=2,
        dim_y=4,
        dim_theta=4,
        n_target_theta=4,
        n_context_init=1,
        n_query_init=200,
        design_scale=1.0,
        outcome_scale=1.0,
        predict_logtheta=False,
        dt=0.05,
        include_time=True,
        normalize_time=True,
        max_T=5,
        use_delta_y=True,
        device="cpu",
        training=True,
        **kwargs,
    ):
        self.name = "DoublePendulum"
        self.training = training

        self.dim_xi = dim_xi
        self.dim_y = dim_y
        self.dim_theta = dim_theta

        self.n_target_data = 0
        self.n_target_theta = n_target_theta

        self.n_context_init = n_context_init
        self.n_query_init = n_query_init

        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.predict_logtheta = predict_logtheta

        self.dt = dt

        self.include_time = include_time
        self.normalize_time = normalize_time
        self.max_T = max_T
        self.use_delta_y = use_delta_y

        self.device = torch.device(device)

        # ALINE input dimension:
        # physical control + previous state + optional time feature
        self.dim_x = dim_xi + dim_y + int(include_time)

        self.simulator = DoublePendulum(
            dt=dt,
            dim_theta=dim_theta,
            dim_xi=dim_xi,
            dim_y=dim_y,
            device=self.device,
        )

    def __repr__(self):
        return (
            f"HiddenDoublePendulum(training={self.training}, "
            f"dim_xi={self.dim_xi}, dim_x={self.dim_x}, dim_y={self.dim_y}, "
            f"dim_theta={self.dim_theta}, include_time={self.include_time}, "
            f"use_delta_y={self.use_delta_y}, dt={self.dt}, "
            f"device={self.device})"
        )

    def normalize_y(self, y):
        if self.outcome_scale is None:
            return y
        return y / self.outcome_scale

    def make_target(self, theta):
        if self.predict_logtheta:
            return torch.log(theta)
        return theta

    def sample_theta(self, batch_size):
        return self.simulator.prior.sample(batch_size)

    def _time_feature(self, batch_size, step, dtype=torch.float32):
        """
        Returns a time feature of shape [B, 1].
        """
        if self.normalize_time:
            denom = max(float(self.max_T), 1.0)
            value = float(step) / denom
        else:
            value = float(step)

        return torch.full(
            (batch_size, 1),
            value,
            device=self.device,
            dtype=dtype,
        )

    def make_aline_x(self, xi, prev_y, step):
        """
        Build augmented ALINE input:

            x = [xi, prev_y, time]

        or, if include_time=False:

            x = [xi, prev_y].
        """
        if xi.ndim != 2:
            raise ValueError(f"xi must have shape [B, dim_xi], got {xi.shape}")

        if prev_y.ndim != 2:
            raise ValueError(f"prev_y must have shape [B, dim_y], got {prev_y.shape}")

        if xi.shape[0] != prev_y.shape[0]:
            raise ValueError(
                f"xi and prev_y must have same batch size, got "
                f"{xi.shape[0]} and {prev_y.shape[0]}"
            )

        if xi.shape[-1] != self.dim_xi:
            raise ValueError(
                f"xi last dim should be dim_xi={self.dim_xi}, got {xi.shape[-1]}"
            )

        if prev_y.shape[-1] != self.dim_y:
            raise ValueError(
                f"prev_y last dim should be dim_y={self.dim_y}, got {prev_y.shape[-1]}"
            )

        pieces = [xi, prev_y]

        if self.include_time:
            time_feat = self._time_feature(
                batch_size=xi.shape[0],
                step=step,
                dtype=xi.dtype,
            )
            pieces.append(time_feat)

        x = torch.cat(pieces, dim=-1)

        if x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"Internal error: x has dim {x.shape[-1]}, expected {self.dim_x}"
            )

        return x

    def sample_query_x(self, batch_size, current_y=None, step=0):
        """
        Sample candidate physical controls and augment them with current state
        and optional time encoding.

        Returns:
            query_x: [B, n_query_init, dim_x]

        Each candidate is:

            [xi_candidate, current_y, time_feature].
        """
        xi_candidates = (
            2.0
            * self.design_scale
            * torch.rand(
                batch_size,
                self.n_query_init,
                self.dim_xi,
                device=self.device,
            )
            - self.design_scale
        )

        if current_y is None:
            current_y = torch.zeros(
                batch_size,
                self.dim_y,
                device=self.device,
                dtype=xi_candidates.dtype,
            )

        if current_y.ndim != 2:
            raise ValueError(
                f"current_y must have shape [B, dim_y], got {current_y.shape}"
            )

        if current_y.shape[0] != batch_size:
            raise ValueError(
                f"current_y batch size should be {batch_size}, "
                f"got {current_y.shape[0]}"
            )

        if current_y.shape[-1] != self.dim_y:
            raise ValueError(
                f"current_y last dim should be dim_y={self.dim_y}, "
                f"got {current_y.shape[-1]}"
            )

        prev_y_rep = current_y[:, None, :].expand(
            batch_size,
            self.n_query_init,
            self.dim_y,
        )

        pieces = [xi_candidates, prev_y_rep]

        if self.include_time:
            time_feat = self._time_feature(
                batch_size=batch_size,
                step=step,
                dtype=xi_candidates.dtype,
            )
            time_feat = time_feat[:, None, :].expand(
                batch_size,
                self.n_query_init,
                1,
            )
            pieces.append(time_feat)

        query_x = torch.cat(pieces, dim=-1)

        if query_x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"query_x has last dim {query_x.shape[-1]}, expected {self.dim_x}"
            )

        return query_x

    def sample_batch(self, batch_size):
        """
        Sample a fresh ALINE batch.

        The initial context is a dummy transition from y0 with xi=0.
        If use_delta_y=True, the initial context_y is zero.
        """
        theta = self.sample_theta(batch_size)  # [B, dim_theta]

        current_y = self.simulator.get_initial_observation(batch_size)
        current_y = current_y.to(self.device)  # [B, dim_y]

        step = 0

        query_x = self.sample_query_x(
            batch_size=batch_size,
            current_y=current_y,
            step=step,
        )

        xi0 = torch.zeros(
            batch_size,
            self.dim_xi,
            device=self.device,
            dtype=current_y.dtype,
        )

        context_x0 = self.make_aline_x(
            xi=xi0,
            prev_y=current_y,
            step=step,
        )  # [B, dim_x]

        context_x = context_x0[:, None, :].repeat(
            1,
            self.n_context_init,
            1,
        )

        if self.use_delta_y:
            context_y0 = torch.zeros_like(current_y)
        else:
            context_y0 = self.normalize_y(current_y)

        context_y = context_y0[:, None, :].repeat(
            1,
            self.n_context_init,
            1,
        )

        batch = AttrDict()
        batch.context_x = context_x
        batch.context_y = context_y
        batch.query_x = query_x

        batch.theta = theta
        batch.target_theta = theta
        batch.target_all = self.make_target(theta)
        batch.n_target_theta = self.n_target_theta

        # True physical state, never normalized.
        batch.current_y = current_y

        batch.t = step

        return batch

    def update_batch(self, batch, idx):
        """
        Update the batch after ALINE selects a query.

        batch.query_x contains augmented candidates:

            [xi_candidate, current_y, time_feature]

        but the simulator only receives the physical control xi_candidate.
        """
        if idx.ndim > 1:
            idx = idx.squeeze(-1)

        B = batch.query_x.shape[0]

        if idx.shape[0] != B:
            raise ValueError(
                f"idx batch size should be {B}, got {idx.shape[0]}"
            )

        arange = torch.arange(B, device=self.device)

        selected_x = batch.query_x[arange, idx]  # [B, dim_x]

        if selected_x.shape[-1] != self.dim_x:
            raise RuntimeError(
                f"selected_x has dim {selected_x.shape[-1]}, expected {self.dim_x}"
            )

        # Extract the true physical control
        xi = selected_x[:, : self.dim_xi]  # [B, 2]

        prev_y = batch.current_y  # [B, 4]

        y_next = self.simulator._sample_outcome(
            theta=batch.theta,
            design=xi,
            last_y=prev_y,
        )  # [B, 4]

        if self.use_delta_y:
            context_y_new = self.normalize_y(y_next - prev_y)
        else:
            context_y_new = self.normalize_y(y_next)

        batch.context_x = torch.cat(
            [batch.context_x, selected_x[:, None, :]],
            dim=1,
        )

        batch.context_y = torch.cat(
            [batch.context_y, context_y_new[:, None, :]],
            dim=1,
        )

        # Keep true physical state unnormalized
        batch.current_y = y_next

        batch.t = int(batch.t) + 1

        batch.query_x = self.sample_query_x(
            batch_size=B,
            current_y=batch.current_y,
            step=batch.t,
        )

        return batch   
