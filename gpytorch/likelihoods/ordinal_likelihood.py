from typing import Any, Dict

import torch
from torch import Tensor

from ..constraints import Positive
from ..distributions import MultivariateNormal
from .likelihood import Likelihood

def inv_probit(x, jitter=1e-3):
    """
    Inverse probit function (standard normal CDF) with jitter for numerical stability.
    
    Args:
        x: Input tensor
        jitter: Small constant to ensure outputs are strictly between 0 and 1
        
    Returns:
        Probabilities between jitter and 1-jitter
    """
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter

class OrdinalLikelihood(Likelihood):
    def __init__(self, bin_edges):
        """
        An ordinal likelihood for regressing over ordinal data.

        The data are integer values from 0 to k, and the user must specify (k-1)
        'bin edges' which define the points at which the labels switch. Let the bin
        edges be [a₀, a₁, ... aₖ₋₁], then the likelihood is

        p(Y=0|F) = ɸ((a₀ - F) / σ)
        p(Y=1|F) = ɸ((a₁ - F) / σ) - ɸ((a₀ - F) / σ)
        p(Y=2|F) = ɸ((a₂ - F) / σ) - ɸ((a₁ - F) / σ)
        ...
        p(Y=K|F) = 1 - ɸ((aₖ₋₁ - F) / σ)

        where ɸ is the cumulative density function of a Gaussian (the inverse probit
        function) and σ is a parameter to be learned.

        A reference is :cite:t:`chu2005gaussian`. 

        :param bin_edges: A tensor of shape (k-1) containing the bin edges.
        """
        super().__init__()
        self.num_bins = len(bin_edges) + 1

        self.register_parameter('bin_edges', torch.nn.Parameter(bin_edges, requires_grad=False))
        self.register_parameter('sigma', torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint('sigma', Positive())

    def forward(self, function_samples: Tensor, *args: Any, data: Dict[str, Tensor] = {}, **kwargs: Any):
        if isinstance(function_samples, MultivariateNormal):
            function_samples = function_samples.sample()
        
        # Compute scaled bin edges
        scaled_edges = self.bin_edges / self.sigma
        scaled_edges_left = torch.cat([scaled_edges, torch.tensor([torch.inf], device=scaled_edges.device)], dim=-1)
        scaled_edges_right = torch.cat([torch.tensor([-torch.inf], device=scaled_edges.device), scaled_edges])
        
        # Calculate cumulative probabilities using standard normal CDF (probit function)
        # These represent P(Y ≤ k | F)
        function_samples = function_samples.unsqueeze(-1)
        scaled_edges_left = scaled_edges_left.reshape(1, 1, -1)
        scaled_edges_right = scaled_edges_right.reshape(1, 1, -1)
        probs = inv_probit(scaled_edges_left - function_samples / self.sigma) - inv_probit(scaled_edges_right - function_samples / self.sigma)
        
        return torch.distributions.Categorical(probs=probs)
