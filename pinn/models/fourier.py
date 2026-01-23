"""
Fourier Feature Encoding for PINNs.

Implements random Fourier features to help neural networks learn
high-frequency functions, as described in:
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
(Tancik et al., NeurIPS 2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class FourierFeatures(nn.Module):
    """
    Random Fourier feature encoding for spatial coordinates.

    Maps low-dimensional inputs (x, y) to high-dimensional features via:
        γ(v) = [cos(2π * B * v), sin(2π * B * v)]

    where B is a matrix of random frequencies sampled from N(0, σ²).

    This helps neural networks learn high-frequency content in the
    potential field, which is common near boundaries and interfaces.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_frequencies: int = 128,
        sigma: float = 4.0,
        include_input: bool = True,
        learnable: bool = False
    ):
        """
        Initialize Fourier feature encoding.

        Args:
            input_dim: Dimension of input coordinates (default: 2 for x, y)
            num_frequencies: Number of frequency components
            sigma: Standard deviation of frequency distribution
            include_input: Whether to concatenate original input
            learnable: Whether frequencies are learnable parameters
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.include_input = include_input
        self.learnable = learnable

        # Output dimension
        self.output_dim = 2 * num_frequencies
        if include_input:
            self.output_dim += input_dim

        # Initialize frequency matrix B
        B = torch.randn(input_dim, num_frequencies) * sigma

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.

        Args:
            x: Input coordinates of shape (..., input_dim)

        Returns:
            Fourier features of shape (..., output_dim)
        """
        # Compute x @ B: (..., num_frequencies)
        x_proj = torch.matmul(x, self.B) * 2 * np.pi

        # Compute sin and cos features
        features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        # Optionally include original input
        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features

    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, num_frequencies={self.num_frequencies}, '
                f'sigma={self.sigma}, include_input={self.include_input}, '
                f'learnable={self.learnable}')


class MultiScaleFourierFeatures(nn.Module):
    """
    Multi-scale Fourier features with different frequency bands.

    Uses multiple sigma values to capture both low and high frequency content.
    This is useful for problems with features at multiple scales.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_frequencies_per_scale: int = 64,
        sigmas: tuple = (1.0, 4.0, 16.0),
        include_input: bool = True
    ):
        """
        Initialize multi-scale Fourier features.

        Args:
            input_dim: Dimension of input coordinates
            num_frequencies_per_scale: Frequencies per scale
            sigmas: Tuple of sigma values for different scales
            include_input: Whether to include original input
        """
        super().__init__()

        self.input_dim = input_dim
        self.sigmas = sigmas
        self.include_input = include_input

        # Create Fourier feature encoders for each scale
        self.encoders = nn.ModuleList([
            FourierFeatures(
                input_dim=input_dim,
                num_frequencies=num_frequencies_per_scale,
                sigma=sigma,
                include_input=False,
                learnable=False
            )
            for sigma in sigmas
        ])

        # Output dimension
        self.output_dim = len(sigmas) * 2 * num_frequencies_per_scale
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale Fourier encoding.

        Args:
            x: Input coordinates of shape (..., input_dim)

        Returns:
            Multi-scale Fourier features
        """
        features = [encoder(x) for encoder in self.encoders]

        if self.include_input:
            features = [x] + features

        return torch.cat(features, dim=-1)


class PositionalEncoding(nn.Module):
    """
    Deterministic positional encoding (sinusoidal).

    Uses fixed frequencies at powers of 2:
        γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^(L-1) π p), cos(2^(L-1) π p)]

    This is the original NeRF-style positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 1,
        num_frequencies: int = 8,
        max_freq_log2: Optional[float] = None,
        include_input: bool = True
    ):
        """
        Initialize positional encoding.

        Args:
            input_dim: Dimension of input
            num_frequencies: Number of frequency octaves
            max_freq_log2: Maximum frequency (log2). If None, uses num_frequencies - 1
            include_input: Whether to include original input
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if max_freq_log2 is None:
            max_freq_log2 = num_frequencies - 1

        # Create frequency bands
        freq_bands = 2 ** torch.linspace(0, max_freq_log2, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)

        # Output dimension
        self.output_dim = 2 * num_frequencies * input_dim
        if include_input:
            self.output_dim += input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: Input of shape (..., input_dim)

        Returns:
            Positional encoding of shape (..., output_dim)
        """
        # x: (..., input_dim)
        # freq_bands: (num_frequencies,)
        # x_scaled: (..., input_dim, num_frequencies)
        x_scaled = x.unsqueeze(-1) * self.freq_bands * np.pi

        # Flatten last two dims and compute sin/cos
        x_scaled = x_scaled.reshape(*x.shape[:-1], -1)
        features = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)

        if self.include_input:
            features = torch.cat([x, features], dim=-1)

        return features
