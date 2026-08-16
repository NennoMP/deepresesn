"""Implementation of the residual reservoir layer of DeepResESN."""

__all__ = ["Reservoir", "ReservoirConfig"]

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.utils._initialization import (
    init_bias,
    init_input_matrix,
    init_ortho_matrix,
    init_recurrent_matrix,
)


@dataclass
class ReservoirConfig:
    """Hyperparameters' configuration for the reservoir.

    Attributes
    ----------
    alpha : float, optional, default=1.0
        Scaling factor for the linear branch (residual connection).
    beta : float, optional, default=1.0
        Scaling factor for the non-linear branch.
    rho : float, optional, default=0.9
        Desired spectral radius of the recurrent weight matrix.
    in_scaling : float, optional, default=1.0
        Scaling factor for the input weight matrix.
    bias_scaling : float, optional, default=0.0
        Scaling factor for the bias vector.
    ortho : str, optional, default="random"
        Type of orthogonal matrix for the temporal skip connections. Options are
        `random`, `cycle`, or `identity`.
    """

    alpha: float = 1.0
    beta: float = 1.0
    rho: float = 0.9
    in_scaling: float = 1.0
    bias_scaling: float = 0.0
    ortho: str = "random"


class Reservoir(nn.Module):
    """The (untrained) residual reservoir layer of DeepResESN.

    Parameters
    ----------
    config : ReservoirConfig
        Configuration for the reservoir layer.
    in_size : int, optional, default=1
        Number of expected input features.
    n_units : int, optional, default=128
        Number of hidden units in the reservoir.

    Attributes
    ----------
    in_matrix : Tensor
        The input weight matrix of the reservoir, of shape (`in_size`, `n_units`).
    recurrent_matrix : Tensor
        The recurrent weight matrix of the reservoir, of shape (`n_units`, `n_units`).
    bias : Tensor
        The bias vector of the reservoir, of shape (`n_units`,).
    ortho_matrix : Tensor
        The orthogonal matrix for the temporal skip connections, of shape (`n_units`,
        `n_units`).
    alpha : Tensor
        Scaling factor for the linear branch (residual connection).
    beta : Tensor
        Scaling factor for the non-linear branch.
    act : Callable[[Tensor], Tensor]
        Element-wise activation function applied to the non-linear branch.
    """

    def __init__(
        self,
        config: ReservoirConfig,
        in_size: int = 1,
        n_units: int = 128,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_size = in_size
        self.n_units = n_units

        in_matrix = init_input_matrix(M=in_size, N=n_units, scaling=config.in_scaling)
        self.register_buffer("in_matrix", in_matrix)

        recurrent_matrix = init_recurrent_matrix(M=n_units, rho=config.rho)
        self.register_buffer("recurrent_matrix", recurrent_matrix)

        bias = init_bias(M=n_units, scaling=config.bias_scaling)
        self.register_buffer("bias", bias)

        # temporal residual connection
        ortho_matrix = init_ortho_matrix(M=n_units, ortho_config=config.ortho)
        self.register_buffer("ortho_matrix", ortho_matrix)

        # scaling factors
        self.register_buffer("alpha", torch.tensor(config.alpha))
        self.register_buffer("beta", torch.tensor(config.beta))

        self.act = torch.tanh

    def _step(self, x: Tensor, h_prev: Tensor) -> Tensor:
        """Performs one time step of the recurrence.

        Parameters
        ----------
        x : Tensor
            Input at the current time step, of shape (`batch`, `in_size`).
        h_prev : Tensor
            Previous hidden state, of shape (`batch`, `n_units`).

        Returns
        -------
        Tensor
            Updated hidden state, of shape (`batch`, `n_units`).
        """
        pre_act = x @ self.in_matrix + h_prev @ self.recurrent_matrix + self.bias
        residual = h_prev @ self.ortho_matrix
        return self.alpha * residual + self.beta * self.act(pre_act)

    def forward(self, x: Tensor, h_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input sequence, of shape (`batch`, `time`, `in_size`).
        h_prev : Tensor, optional, default=None
            Initial state, of shape (`batch`, `n_units`). If None, defaults to a zero
            tensor.

        Returns
        -------
        states : Tensor
            Hidden states at every time step, of shape (`batch`, `time`, `n_units`).
        last_state : Tensor
            Hidden state at the final time step, of shape (`batch`, `n_units`).
        """
        batch, n_steps = x.shape[0], x.shape[1]

        if h_prev is None:
            h_prev = torch.zeros(batch, self.n_units, device=x.device)

        states = torch.empty(batch, n_steps, self.n_units, device=x.device)
        for t in range(n_steps):
            h_prev = self._step(x[:, t], h_prev)
            states[:, t] = h_prev

        return (states, h_prev)
