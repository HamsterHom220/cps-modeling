"""
Normalization utilities for PINN inputs and outputs.

Handles normalization of:
- 8D input parameters
- Spatial coordinates (x, y)
- Time
- Potential field φ
- Conductivity field σ
- 26 scalar metrics
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class NormalizationStats:
    """Statistics for min-max or standardization normalization."""
    min_val: torch.Tensor
    max_val: torch.Tensor
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None


class Normalizer:
    """
    Handles normalization of all inputs and outputs for PINN.

    Uses min-max normalization to [0, 1] for bounded quantities
    and standardization for unbounded quantities.
    """

    # Parameter ranges from dataset (empirically determined)
    PARAM_RANGES = {
        'R_sigma': (3.0, 8.0),          # Soil resistivity [Ω·m]
        'roughness': (0.1, 0.5),         # Surface roughness [μm]
        'coating_quality': (0.7, 0.95),  # Coating quality [0-1]
        'pH': (6.5, 8.0),                # pH
        'V_app': (3.0, 7.0),             # Applied voltage [V]
        'humidity': (0.3, 0.8),          # Humidity [0-1]
        'age': (0.0, 30.0),              # Age [years]
        'anode_efficiency': (0.9, 0.95)  # Anode efficiency [0-1]
    }

    PARAM_NAMES = ['R_sigma', 'roughness', 'coating_quality', 'pH',
                   'V_app', 'humidity', 'age', 'anode_efficiency']

    # Domain geometry
    DOMAIN_X = (0.0, 20.0)  # [m]
    DOMAIN_Y = (0.0, 8.0)   # [m]

    # Time range
    TIME_RANGE = (0.0, 30.0)  # [years]

    # Field ranges (approximate, from FEM simulations)
    PHI_RANGE = (-1.5, 1.5)     # Potential [V]
    SIGMA_RANGE = (0.01, 0.5)   # Conductivity [S/m]

    # Scalar metric ranges (empirically determined)
    SCALAR_RANGES = {
        'coverage': (0.0, 100.0),
        'avg_potential': (-1.2, 1.1),
        'min_potential': (-1.2, 1.1),
        'max_potential': (-1.2, 1.1),
        'current': (0.0, 100.0),
        'current_density': (0.0, 10.0),
        'corrosion_rate': (0.0, 0.5),
        'anode_resistance': (0.0, 100.0),
        'pipe_resistance': (0.0, 100.0),
        'coating_resistance': (100.0, 10000.0),
        'voltage_drop': (0.0, 5.0),
    }

    def __init__(self, device: str = 'cpu'):
        """
        Initialize normalizer with default ranges.

        Args:
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.device = device
        self._init_param_stats()
        self._init_coord_stats()
        self._init_field_stats()
        self._init_scalar_stats()

    def _init_param_stats(self):
        """Initialize parameter normalization statistics."""
        min_vals = torch.tensor([self.PARAM_RANGES[n][0] for n in self.PARAM_NAMES],
                                dtype=torch.float32, device=self.device)
        max_vals = torch.tensor([self.PARAM_RANGES[n][1] for n in self.PARAM_NAMES],
                                dtype=torch.float32, device=self.device)
        self.param_stats = NormalizationStats(min_val=min_vals, max_val=max_vals)

    def _init_coord_stats(self):
        """Initialize coordinate normalization statistics."""
        self.x_stats = NormalizationStats(
            min_val=torch.tensor(self.DOMAIN_X[0], dtype=torch.float32, device=self.device),
            max_val=torch.tensor(self.DOMAIN_X[1], dtype=torch.float32, device=self.device)
        )
        self.y_stats = NormalizationStats(
            min_val=torch.tensor(self.DOMAIN_Y[0], dtype=torch.float32, device=self.device),
            max_val=torch.tensor(self.DOMAIN_Y[1], dtype=torch.float32, device=self.device)
        )
        self.t_stats = NormalizationStats(
            min_val=torch.tensor(self.TIME_RANGE[0], dtype=torch.float32, device=self.device),
            max_val=torch.tensor(self.TIME_RANGE[1], dtype=torch.float32, device=self.device)
        )

    def _init_field_stats(self):
        """Initialize field normalization statistics."""
        self.phi_stats = NormalizationStats(
            min_val=torch.tensor(self.PHI_RANGE[0], dtype=torch.float32, device=self.device),
            max_val=torch.tensor(self.PHI_RANGE[1], dtype=torch.float32, device=self.device)
        )
        self.sigma_stats = NormalizationStats(
            min_val=torch.tensor(self.SIGMA_RANGE[0], dtype=torch.float32, device=self.device),
            max_val=torch.tensor(self.SIGMA_RANGE[1], dtype=torch.float32, device=self.device)
        )

    def _init_scalar_stats(self):
        """Initialize scalar metric normalization statistics."""
        self.scalar_stats = {}
        for name, (min_v, max_v) in self.SCALAR_RANGES.items():
            self.scalar_stats[name] = NormalizationStats(
                min_val=torch.tensor(min_v, dtype=torch.float32, device=self.device),
                max_val=torch.tensor(max_v, dtype=torch.float32, device=self.device)
            )

    @staticmethod
    def _normalize_minmax(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
        """Min-max normalize to [0, 1]."""
        return (x - stats.min_val) / (stats.max_val - stats.min_val + 1e-8)

    @staticmethod
    def _denormalize_minmax(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
        """Inverse of min-max normalization."""
        return x * (stats.max_val - stats.min_val) + stats.min_val

    def normalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        Normalize 8D input parameters to [0, 1].

        Args:
            params: Parameter tensor of shape (..., 8)

        Returns:
            Normalized parameters of same shape
        """
        return self._normalize_minmax(params, self.param_stats)

    def denormalize_params(self, params: torch.Tensor) -> torch.Tensor:
        """Denormalize parameters from [0, 1] to original range."""
        return self._denormalize_minmax(params, self.param_stats)

    def normalize_coords(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize spatial coordinates to [0, 1].

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            Tuple of normalized (x, y)
        """
        x_norm = self._normalize_minmax(x, self.x_stats)
        y_norm = self._normalize_minmax(y, self.y_stats)
        return x_norm, y_norm

    def denormalize_coords(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Denormalize coordinates from [0, 1] to original range."""
        x_denorm = self._denormalize_minmax(x, self.x_stats)
        y_denorm = self._denormalize_minmax(y, self.y_stats)
        return x_denorm, y_denorm

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize time to [0, 1]."""
        return self._normalize_minmax(t, self.t_stats)

    def denormalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Denormalize time from [0, 1] to original range."""
        return self._denormalize_minmax(t, self.t_stats)

    def normalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        """Normalize potential field to [0, 1]."""
        return self._normalize_minmax(phi, self.phi_stats)

    def denormalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        """Denormalize potential field from [0, 1] to original range."""
        return self._denormalize_minmax(phi, self.phi_stats)

    def normalize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Normalize conductivity field to [0, 1]."""
        return self._normalize_minmax(sigma, self.sigma_stats)

    def denormalize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Denormalize conductivity field from [0, 1] to original range."""
        return self._denormalize_minmax(sigma, self.sigma_stats)

    def normalize_scalars(self, scalars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize scalar metrics."""
        normalized = {}
        for name, value in scalars.items():
            if name in self.scalar_stats:
                normalized[name] = self._normalize_minmax(value, self.scalar_stats[name])
            else:
                # Unknown metric, pass through
                normalized[name] = value
        return normalized

    def denormalize_scalars(self, scalars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize scalar metrics."""
        denormalized = {}
        for name, value in scalars.items():
            if name in self.scalar_stats:
                denormalized[name] = self._denormalize_minmax(value, self.scalar_stats[name])
            else:
                denormalized[name] = value
        return denormalized

    def fit_from_data(self,
                      params: Optional[np.ndarray] = None,
                      phi: Optional[np.ndarray] = None,
                      sigma: Optional[np.ndarray] = None,
                      scalars: Optional[Dict[str, np.ndarray]] = None):
        """
        Update normalization statistics from actual data.

        This can be used to compute more accurate ranges from the training set.

        Args:
            params: Array of parameters (N, 8)
            phi: Array of potential fields (N, H, W)
            sigma: Array of conductivity fields (N, H, W)
            scalars: Dict of scalar arrays
        """
        if params is not None:
            min_vals = torch.tensor(params.min(axis=0), dtype=torch.float32, device=self.device)
            max_vals = torch.tensor(params.max(axis=0), dtype=torch.float32, device=self.device)
            # Add small margin
            margin = (max_vals - min_vals) * 0.05
            self.param_stats = NormalizationStats(
                min_val=min_vals - margin,
                max_val=max_vals + margin
            )

        if phi is not None:
            self.phi_stats = NormalizationStats(
                min_val=torch.tensor(phi.min() - 0.1, dtype=torch.float32, device=self.device),
                max_val=torch.tensor(phi.max() + 0.1, dtype=torch.float32, device=self.device)
            )

        if sigma is not None:
            self.sigma_stats = NormalizationStats(
                min_val=torch.tensor(sigma.min() * 0.9, dtype=torch.float32, device=self.device),
                max_val=torch.tensor(sigma.max() * 1.1, dtype=torch.float32, device=self.device)
            )

        if scalars is not None:
            for name, values in scalars.items():
                self.scalar_stats[name] = NormalizationStats(
                    min_val=torch.tensor(values.min(), dtype=torch.float32, device=self.device),
                    max_val=torch.tensor(values.max(), dtype=torch.float32, device=self.device)
                )

    def to(self, device: str):
        """Move all statistics to specified device."""
        self.device = device

        def move_stats(stats: NormalizationStats) -> NormalizationStats:
            return NormalizationStats(
                min_val=stats.min_val.to(device),
                max_val=stats.max_val.to(device),
                mean=stats.mean.to(device) if stats.mean is not None else None,
                std=stats.std.to(device) if stats.std is not None else None
            )

        self.param_stats = move_stats(self.param_stats)
        self.x_stats = move_stats(self.x_stats)
        self.y_stats = move_stats(self.y_stats)
        self.t_stats = move_stats(self.t_stats)
        self.phi_stats = move_stats(self.phi_stats)
        self.sigma_stats = move_stats(self.sigma_stats)

        for name in self.scalar_stats:
            self.scalar_stats[name] = move_stats(self.scalar_stats[name])

        return self

    def state_dict(self) -> Dict:
        """Return state dict for saving."""
        return {
            'param_stats': self.param_stats,
            'x_stats': self.x_stats,
            'y_stats': self.y_stats,
            't_stats': self.t_stats,
            'phi_stats': self.phi_stats,
            'sigma_stats': self.sigma_stats,
            'scalar_stats': self.scalar_stats,
            'device': self.device
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.param_stats = state_dict['param_stats']
        self.x_stats = state_dict['x_stats']
        self.y_stats = state_dict['y_stats']
        self.t_stats = state_dict['t_stats']
        self.phi_stats = state_dict['phi_stats']
        self.sigma_stats = state_dict['sigma_stats']
        self.scalar_stats = state_dict['scalar_stats']
        self.device = state_dict.get('device', 'cpu')
