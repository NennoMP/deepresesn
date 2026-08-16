__all__ = [
    "init_bias",
    "init_input_matrix",
    "init_ortho_matrix",
    "init_recurrent_matrix",
]

import numpy as np
import torch
from torch import Tensor


def init_bias(M: int, scaling: float) -> Tensor:
    """
    Initialize a (`M`,) real-valued bias vector, whose entries are sampled uniformly
    from [-`scaling`, `scaling`].

    Parameters
    ----------
    M : int
        Size of the bias vector.
    scaling : float
        Scaling factor for the bias vector.

    Returns
    -------
    Tensor
        A (`M`,) real-valued tensor containing the initialized bias vector
    """
    return torch.zeros(M).uniform_(-scaling, scaling)


def init_input_matrix(M: int, N: int, scaling: float) -> Tensor:
    """
    Initialize a (`M`, `N`) real-valued dense matrix, whose entries are sampled
    uniformly from [-`scaling`, `scaling`].

    Parameters
    ----------
    M : int
        Number of rows (input dimension).
    N : int
        Number of columns (output dimension).
    scaling : float
        Scaling factor for the kernel entries.

    Returns
    -------
    Tensor
        A (`M`, `N`) real-valued tensor containing the initialized matrix.
    """
    return torch.zeros((M, N)).uniform_(-scaling, scaling)


def init_ortho_matrix(M: int, ortho_config: str) -> Tensor:
    """
    Initialize a (`M`, `M`) real-valued orthogonal matrix, to be used as the
    transformation in temporal residual connections.

    Supported orthogonal initializations include:
    - `random`: a random orthogonal matrix obtained via QR decomposition (`R` in the
    paper)
    - `cycle`: a cyclic orthogonal matrix (`C` in the paper)
    - `identity`: the identity matrix (`I` in the paper)

    Parameters
    ----------
    M : int
        Number of rows and columns (square matrix).
    ortho_config : str
        Type of orthogonal initialization. Options are `random`, `cycle`, or `identity`.

    Returns
    -------
    Tensor
        A (`M`, `M`) real-valued tensor containing the specified orthogonal matrix.

    Raises
    ------
    ValueError
        If `ortho_config` is not one of `random`, `cycle`, or `identity`.
    """
    if ortho_config == "random":
        O, _ = np.linalg.qr(2 * np.random.rand(M, M) - 1)
    elif ortho_config == "cycle":
        O = np.zeros((M, M))
        O[0, M - 1] = 1
        O[np.arange(1, M), np.arange(M - 1)] = 1
    elif ortho_config == "identity":
        O = np.eye(M, M)
    else:
        raise ValueError(
            f"Invalid skip option: {ortho_config}. "
            f"Options are `random`, `cycle`, or `identity`."
        )

    return torch.from_numpy(O).to(torch.get_default_dtype())


def init_recurrent_matrix(M: int, rho: float) -> torch.Tensor:
    """
    Initialize a (`M`, `M`) real-valued dense matrix, whose entries are sampled
    uniformly in the range [-1, 1] and then rescaled to have spectral radius
    *approximately* `rho`.

    The rescaling relies on the circular law: for a matrix with i.i.d. zero-mean
    entries of standard deviation `sigma`, the spectral radius concentrates around
    `sigma * sqrt(M)`. Since `U(-1, 1)` has standard deviation `1 / sqrt(3)`, the
    entries are scaled by `rho * sqrt(3) / sqrt(M)`. This is an asymptotic result, so
    the realised spectral radius deviates from `rho` for small `M` (it is within a few
    percent for `M` in the hundreds).

    Parameters
    ----------
    M : int
        Number of rows and columns (square matrix).
    rho : float
        Desired (approximate) spectral radius.

    Returns
    -------
    Tensor
        A (`M`, `M`) real-valued tensor containing the initialized matrix.
    """
    W = torch.zeros((M, M)).uniform_(-1, 1)
    W *= (rho / np.sqrt(M)) * (6 / np.sqrt(12))  # sqrt(3) = 6 / sqrt(12)
    return W
