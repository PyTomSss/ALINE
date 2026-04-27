import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Any, List, Tuple, Optional
from attrdictionary import AttrDict


class AcquisitionHead(nn.Module):
    """
    Acquisition head that predicts the acquisition scores for the query data
    """
    def __init__(self, dim_embedding: int, dim_feedforward: int, time_token: bool, **kwargs: Any) -> None:
        """
        Initialize the acquisition head
        
        Args:
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            time_token: Whether to include a time token in the embeddings
        """
        super().__init__()
        
        if time_token:
            dim_embedding += 1  # add time token to embedding

        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
            nn.Flatten(start_dim=-2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the acquisition head
        
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            acquisition_scores: Acquisition scores [B, N]
        """
        return self.predictor(z)
    


class ContinuousAcquisitionHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, dim_x: int, time_token: bool, **kwargs: Any) -> None:
        super().__init__()

        if time_token:
            dim_embedding += 1  # add time token to embedding
        
        # Network for mean
        self.mean_predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_x)
        )
        
        # Network for log_std
        self.log_std_predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_x)
        )
    
    def forward(self, z):
        """
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            mean: Mean of Gaussian policy [B, dim_x]
            std: Standard deviation of Gaussian policy [B, dim_x]
        """
        mean = self.mean_predictor(z)                   # [B, n_query=1, dim_x]
        log_std = self.log_std_predictor(z)             # [B, n_query=1, dim_x]
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        
        return mean, std
    


class ValueHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, **kwargs: Any) -> None:
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )

        # value for zero context
        self.empty_value = nn.Parameter(torch.zeros(1))

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Embedding feature tensor [B, t, dim_embedding]
        Returns:
            value: Values [B]
        """
        if z.shape[1] == 0:
            z = self.empty_value.expand(z.shape[0], 1)  # [B, 1]
        else:
            # z = z.detach()                              # detach to prevent gradient flow
            z = self.predictor(z).squeeze(-1)           # [B, t]

        return z.mean(1)



class GMMTargetHead(nn.Module):
    """
    Target head predicting a univariate GMM per target token.

    Input:
        z_target: [B, n_target, dim_embedding]

    Output:
        mixture_means:   [B, n_target, num_components]
        mixture_stds:    [B, n_target, num_components]
        mixture_weights: [B, n_target, num_components]
    """

    def __init__(
        self,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int,
        single_head: bool = False,
        std_min: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward

        # Kept for backward compatibility, but no longer used as an output multiplier.
        self.dim_y = dim_y

        self.single_head = single_head
        self.num_components = num_components
        self.std_min = std_min

        self.heads = self.initialize_head(
            dim_embedding=self.dim_embedding,
            dim_feedforward=self.dim_feedforward,
            single_head=self.single_head,
            num_components=self.num_components,
        )

    def forward(self, batch: AttrDict, z_target: torch.Tensor) -> AttrDict:
        """
        z_target: [B, n_target, dim_embedding]
        """
        if self.single_head:
            output = self.heads(z_target)  # [B, n_target, 3 * C] or [B, n_target, 2] if C=1

            if self.num_components == 1:
                raw_mean, raw_std = torch.chunk(output, 2, dim=-1)
                raw_weights = torch.zeros_like(raw_mean)
            else:
                raw_mean, raw_std, raw_weights = torch.chunk(output, 3, dim=-1)

        else:
            outputs = [head(z_target) for head in self.heads]
            raw_mean, raw_std, raw_weights = self._map_raw_output(outputs)

        mean = raw_mean
        std = F.softplus(raw_std) + self.std_min
        weights = F.softmax(raw_weights, dim=-1)

        return AttrDict(
            mixture_means=mean,
            mixture_stds=std,
            mixture_weights=weights,
        )

    def initialize_head(
        self,
        dim_embedding: int,
        dim_feedforward: int,
        single_head: bool,
        num_components: int,
    ) -> nn.Module:
        if single_head:
            if num_components == 1:
                output_dim = 2
            else:
                output_dim = 3 * num_components

            model = nn.Sequential(
                nn.Linear(dim_embedding, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, output_dim),
            )

        else:
            model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim_embedding, dim_feedforward),
                        nn.ReLU(),
                        nn.Linear(dim_feedforward, 3),
                    )
                    for _ in range(num_components)
                ]
            )

        return model

    @staticmethod
    def _map_raw_output(outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        outputs: list of C tensors, each [B, n_target, 3]

        Returns:
            raw_mean:    [B, n_target, C]
            raw_std:     [B, n_target, C]
            raw_weights: [B, n_target, C]
        """
        stacked = torch.stack(outputs, dim=-2)  # [B, n_target, C, 3]

        raw_mean = stacked[..., 0]      # [B, n_target, C]
        raw_std = stacked[..., 1]       # [B, n_target, C]
        raw_weights = stacked[..., 2]   # [B, n_target, C]

        return raw_mean, raw_std, raw_weights



class OutputHead(nn.Module):
    """
    Combined head that processes batches and routes to acquisition and target heads.
    Similar to DPTNP's forward method, it splits input into query and posterior parts.
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int = 10,
        single_head: bool = False,
        std_min: float = 1e-4,
        value_head: bool = False,
        time_token: bool = False,
        dim_target: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_target = dim_y if dim_target is None else dim_target

        self.time_token = time_token
        
        # Acquisition head for design selection
        self.acquisition_head = AcquisitionHead(
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            time_token=time_token,
        )
        
        # Target head for posterior prediction
        self.target_head = GMMTargetHead(
            #dim_y=dim_y,
            dim_y=self.dim_target, 
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            single_head=single_head,
            std_min=std_min
        )

        # Value head for value prediction
        self.value_head = value_head
        if value_head:
            self.value_head = ValueHead(
                dim_embedding=dim_embedding,
                dim_feedforward=dim_feedforward
            )
    
    def forward(self, batch: AttrDict, z: torch.Tensor) -> AttrDict:
        """
        Process batch by splitting into context, query and target parts
        
        Args:
            batch: Batch containing context and target tasks
            z: Embedding tensor [B, N, dim_embedding] where N = n_context + n_query + n_target
                Context embeddings are on the left, query in the middle, target on the right
            
        Returns:
            outs: AttrDict containing acquisition and posterior prediction results
        """
        batch_size = z.shape[0]
        
        # Get dimensions from batch
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]
        
        # Extract query and target embeddings
        z_query = z[:, n_context:n_context+n_query]
        z_target = z[:, n_context+n_query:]  # embeddings of target data + target theta
        
        # Use acquisition head for query selection (design point)
        if self.time_token:
            time_info = batch.t.unsqueeze(1).unsqueeze(1).expand(batch_size, n_query, 1)  # [B, n_query, 1]
            z_query_with_time = torch.cat([z_query, time_info], dim=-1)
            zt = self.acquisition_head(z_query_with_time)  # [B, n_query]
        else:
            zt = self.acquisition_head(z_query)  # [B, n_query]
        
        # Select design with the highest probability during inference, sample during training
        if self.training:
            # Choose design with probabilities zt
            m_design = Categorical(zt)
            idx_next = m_design.sample()  # [B]
            log_prob = m_design.log_prob(idx_next)
        else:
            # Choose design with the largest probability
            log_prob, idx_next = torch.max(zt, -1)
            log_prob = torch.log(log_prob)
            
        
        # Get the selected design point
        idx_next = idx_next.unsqueeze(1) # [B, 1]
        
        # Use target head for posterior prediction
        posterior_out = self.target_head(batch, z_target)
        posterior_out_query = self.target_head(batch, z_query)  # For ACE uncertainty sampling baseline
        
        # Combine results
        # Value prediction
        if self.value_head:
            z_context = z[:, :n_context]
            value = self.value_head(z_context)
            outs = AttrDict(
                posterior_out_query=posterior_out_query,
                posterior_out=posterior_out,
                design_out=AttrDict(
                    idx=idx_next,       # [B, 1]
                    log_prob=log_prob,  # [B]
                    zt=zt               # [B, N_query]
                ),
                value=value             # [B]
            )
        else:
            outs = AttrDict(
                posterior_out_query=posterior_out_query,
                posterior_out=posterior_out,
                design_out=AttrDict(
                    idx=idx_next,       # [B, 1]
                    log_prob=log_prob,  # [B]
                    zt=zt               # [B, N_query]
                ),
            )
        return outs
    


class ContinuousOutputHead(nn.Module):
    """
    Combined head that processes batches and routes to acquisition and target heads.
    Similar to DPTNP's forward method, it splits input into query and posterior parts.
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int = 10,
        single_head: bool = False,
        std_min: float = 1e-4,
        value_head: bool = False,
        time_token: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.time_token = time_token
        
        # Acquisition head for design selection
        self.acquisition_head = ContinuousAcquisitionHead(
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            dim_x=dim_x,
            time_token=time_token,
        )
        
        # Target head for posterior prediction
        self.target_head = GMMTargetHead(
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            single_head=single_head,
            std_min=std_min
        )
    
    def forward(self, batch: AttrDict, z: torch.Tensor) -> AttrDict:
        """
        Process batch by splitting into context, query and target parts
        
        Args:
            batch: Batch containing context and target tasks
            z: Embedding tensor [B, N, dim_embedding] where N = n_context + n_query + n_target
                Context embeddings are on the left, query in the middle, target on the right
            
        Returns:
            outs: AttrDict containing acquisition and posterior prediction results
        """
        batch_size = z.shape[0]
        
        # Get dimensions from batch
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]
        
        # Extract query and target embeddings
        z_query = z[:, n_context:n_context+n_query]
        z_target = z[:, n_context+n_query:]  # embeddings of target data + target theta
        
        # Use acquisition head for query selection (design point)
        if self.time_token:
            time_info = batch.t.unsqueeze(1).unsqueeze(1).expand(batch_size, n_query, 1)  # [B, n_query, 1]
            z_query_with_time = torch.cat([z_query, time_info], dim=-1)
            mean, std = self.acquisition_head(z_query_with_time)  # [B, n_query]
        else:
            mean, std = self.acquisition_head(z_query)  # [B, n_query]
        
        # Select design with the highest probability during inference, sample during training
        if self.training:
            # Choose design with probabilities normal(mean, std)
            x_query = Normal(mean, std).sample()
        else:
            # Choose design with the largest probability
            # Optional todo: multiple designs in a step
            x_query = mean

        log_prob = Normal(mean, std).log_prob(x_query)  # [B, n_query=1, dim_x]
        log_prob = log_prob.sum(-1).squeeze(-1)         # [B]
        
        # Use target head for posterior prediction
        posterior_out = self.target_head(batch, z_target)
        posterior_out_query = self.target_head(batch, z_query)  # For ACE uncertainty sampling baseline
        
        # Combine results
        outs = AttrDict(
            posterior_out_query=posterior_out_query,
            posterior_out=posterior_out,
            design_out=AttrDict(
                xi=x_query,       # [B, n_query=1, dim_x]
                log_prob=log_prob,  # [B]
                mean=mean,
                std=std
            ),
        )
        return outs

