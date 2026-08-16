"""Implementation of Deep Residual Echo State Network (DeepResESN).

Code adapted from [1]_ and [2]_, where Deep Echo State Networks and Residual Echo
State Networks were respectively introduced.

References
----------
.. [1] https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
.. [2] https://github.com/andreaceni/ResESN
"""

__all__ = ["DeepResESN"]

import warnings

import torch
import torch.nn as nn
from torch import Tensor

from src.deepresesn._reservoir import Reservoir, ReservoirConfig


class DeepResESN(nn.Module):
    """
    DeepResESN consists of multiple untrained reservoir layers, each based on
    orthogonal residual connections along the temporal dimensions. The first layer is
    configured with the given hyperparameters, while deeper layers can be configured
    independently with the `inter_*` arguments. The returned states can either be the
    last layer's states or the concatenation of all layers' states.

    Parameters
    ----------
    concat : bool, optional, default=False
        Whether to concatenate the hidden states of all layers or return only the last
        layer's hidden states.
    n_layers : int, optional, default=1
        Number of reservoir layers.
    in_size : int, optional, default=1
        Number of expected input features.
    n_units : int, optional, default=100
        Number of hidden units in the reservoir.
    reservoir_config, inter_reservoir_config : ReservoirConfig, optional, default=None
        Configuration for the first reservoir layer (l=1) and subsequent reservoir
        layers (l > 1). If `reservoir_config` is None, the default configuration is
        used. If `inter_reservoir_config` is None, the same configuration as
        `reservoir_config` is used.

    Any additional keyword argument is ignored, with a warning.

    Attributes
    ----------
    layers : nn.Sequential
        The stacked reservoir layers.
    n_units : int
        Dimensionality of the returned states.
    layers_units : int
        Number of units of every layer after the first.
    first_layer_units : int
        Number of units of the first layer.
    """

    def __init__(
        self,
        concat: bool = False,
        n_layers: int = 1,
        in_size: int = 1,
        n_units: int = 100,
        reservoir_config: ReservoirConfig | None = None,
        inter_reservoir_config: ReservoirConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Raises
        ------
        AssertionError
            If `n_layers` is not greater than 0.
        AssertionError
            If `in_size` is not greater than 0.
        AssertionError
            If `n_units` is not greater than 0.
        AssertionError
            If `concat` is True and `n_units` is smaller than `n_layers`, as the units
            could not be distributed across the layers.
        """
        super().__init__()
        assert n_layers > 0, "`n_layers` must be greater than 0."
        assert in_size > 0, "`in_size` must be greater than 0."
        assert n_units > 0, "`n_units` must be greater than 0."
        assert not concat or n_units >= n_layers, (
            f"When `concat` is True, `n_units` ({n_units}) must be at least "
            f"`n_layers` ({n_layers}), otherwise some layers would have no units."
        )

        if reservoir_config is None:
            reservoir_config = ReservoirConfig()

        if kwargs:
            warnings.warn(
                f"Ignoring unknown keyword arguments: {sorted(kwargs)}.",
                stacklevel=2,
            )

        self.in_size = in_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.concat = concat

        self.reservoir_config = reservoir_config
        self.inter_reservoir_config = reservoir_config
        if inter_reservoir_config is not None:
            self.inter_reservoir_config = inter_reservoir_config

        # if `concat == True` the number of reservoir units `n_units` is evenly divided
        # among the layers; if an even distribution is not possible, the extra units
        # are allocated to the first layer
        self.layers_units = self.first_layer_units = n_units
        if concat:
            self.layers_units = n_units // n_layers
            self.first_layer_units = self.layers_units + n_units % n_layers

        self.layers = self._make_layers()

    def _make_layers(self) -> nn.Sequential:
        """Initialize the reservoir layers.

        Returns
        -------
        nn.Sequential
            A sequential container of reservoir layers.
        """
        layers = [
            Reservoir(
                config=self.reservoir_config,
                in_size=self.in_size,
                n_units=self.first_layer_units,
            ),
        ]

        # subsequent layers
        h_dim = self.first_layer_units
        for _ in range(1, self.n_layers):
            layers.append(
                Reservoir(
                    config=self.inter_reservoir_config,
                    in_size=h_dim,
                    n_units=self.layers_units,
                ),
            )
            h_dim = self.layers_units

        return nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Real-valued input tensor of shape (`batch`, `time`, `in_size`).

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple of real-valued tensors containing the collection of hidden states
            at every time step and the hidden state at the final time step,
            respectively. If `concat` is False these are the states of the last layer
            only, otherwise they are the states of all layers concatenated along the
            feature dimension. In both cases the shape of the first tensor is (`batch`,
            `time`, `n_units`) and the shape of the second tensor is (`batch`,
            `n_units`).
        """
        states, last_states = [], []

        for layer in self.layers:
            x, last_state = layer(x)
            states.append(x)
            last_states.append(last_state)

        if self.concat:  # concatenated hidden states from all layers
            return torch.cat(states, dim=-1), torch.cat(last_states, dim=-1)
        return states[-1], last_states[-1]  # hidden states from the last layer
