"""
Physics-Informed Neural Network for Cathodic Protection System.

Main PINN class that combines all encoder networks and supports
both field and scalar predictions with physics-based loss computation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from .encoders import (
    ParameterEncoder,
    CoordinateEncoder,
    TimeEncoder,
    FusionNetwork,
    FieldHead,
    ScalarHead
)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for CP System Prediction.

    Architecture (Modified DeepONet):
    ```
    Parameters (8D) ──► [Parameter Encoder] ──┐
                                              │
    Coordinates (x,y) ► [Fourier + Trunk] ────┼──► [Fusion] ──► [Field Head] ──► φ(x,y)
                                              │            │
    Time (t) ──────────► [Time Encoder] ──────┘            └──► [Scalar Head] ► metrics
    ```

    Features:
    - Fourier feature encoding for spatial coordinates
    - Positional encoding for time
    - Skip connections in fusion network
    - Supports autograd for PDE residual computation
    - Outputs both potential field and scalar metrics
    """

    # Scalar metric names (in order, matching HDF5 dataset)
    SCALAR_NAMES = [
        'V_app', 'anode_efficiency', 'anode_points', 'anode_potential',
        'anode_resistance', 'avg_potential', 'coating_quality', 'coating_resistance',
        'corrosion_rate', 'coverage', 'current', 'current_conservation_error',
        'current_density', 'flux_anode_current', 'flux_anode_density',
        'flux_pipe_current', 'flux_pipe_density', 'max_potential', 'min_potential',
        'newton_converged', 'newton_iterations', 'pipe_points', 'pipe_resistance',
        'soil_resistivity', 'std_potential', 'voltage_drop'
    ]

    def __init__(
        self,
        # Parameter encoder
        param_input_dim: int = 8,
        param_hidden_dims: List[int] = [64, 128, 256],
        param_output_dim: int = 128,
        # Coordinate encoder
        coord_input_dim: int = 2,
        num_fourier_features: int = 128,
        fourier_sigma: float = 4.0,
        coord_hidden_dims: List[int] = [128, 256],
        coord_output_dim: int = 128,
        # Time encoder
        time_num_frequencies: int = 8,
        time_hidden_dims: List[int] = [32],
        time_output_dim: int = 64,
        # Fusion network
        fusion_hidden_dims: List[int] = [256, 256],
        fusion_output_dim: int = 128,
        fusion_use_skip: bool = True,
        # Field head
        field_hidden_dim: int = 64,
        # Scalar head
        scalar_hidden_dims: List[int] = [128, 64],
        num_scalars: int = 26,
        # General
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        """
        Initialize PINN.

        Args:
            param_input_dim: Number of input parameters
            param_hidden_dims: Parameter encoder hidden dimensions
            param_output_dim: Parameter encoder output dimension
            coord_input_dim: Coordinate dimension (2 for x, y)
            num_fourier_features: Number of Fourier frequency components
            fourier_sigma: Std of Fourier frequencies
            coord_hidden_dims: Coordinate encoder hidden dimensions
            coord_output_dim: Coordinate encoder output dimension
            time_num_frequencies: Time positional encoding frequencies
            time_hidden_dims: Time encoder hidden dimensions
            time_output_dim: Time encoder output dimension
            fusion_hidden_dims: Fusion network hidden dimensions
            fusion_output_dim: Fusion network output dimension
            fusion_use_skip: Use skip connections in fusion
            field_hidden_dim: Field head hidden dimension
            scalar_hidden_dims: Scalar head hidden dimensions
            num_scalars: Number of scalar outputs
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        # Store config
        self.config = {
            'param_input_dim': param_input_dim,
            'param_hidden_dims': param_hidden_dims,
            'param_output_dim': param_output_dim,
            'coord_input_dim': coord_input_dim,
            'num_fourier_features': num_fourier_features,
            'fourier_sigma': fourier_sigma,
            'coord_hidden_dims': coord_hidden_dims,
            'coord_output_dim': coord_output_dim,
            'time_num_frequencies': time_num_frequencies,
            'time_hidden_dims': time_hidden_dims,
            'time_output_dim': time_output_dim,
            'fusion_hidden_dims': fusion_hidden_dims,
            'fusion_output_dim': fusion_output_dim,
            'fusion_use_skip': fusion_use_skip,
            'field_hidden_dim': field_hidden_dim,
            'scalar_hidden_dims': scalar_hidden_dims,
            'num_scalars': num_scalars,
            'activation': activation,
            'dropout': dropout
        }

        # Parameter encoder
        self.param_encoder = ParameterEncoder(
            input_dim=param_input_dim,
            hidden_dims=param_hidden_dims,
            output_dim=param_output_dim,
            activation=activation,
            dropout=dropout
        )

        # Coordinate encoder
        self.coord_encoder = CoordinateEncoder(
            input_dim=coord_input_dim,
            num_fourier_features=num_fourier_features,
            fourier_sigma=fourier_sigma,
            hidden_dims=coord_hidden_dims,
            output_dim=coord_output_dim,
            activation='tanh'  # Tanh for smooth coordinate embeddings
        )

        # Time encoder
        self.time_encoder = TimeEncoder(
            num_frequencies=time_num_frequencies,
            hidden_dims=time_hidden_dims,
            output_dim=time_output_dim,
            activation=activation
        )

        # Fusion network
        self.fusion = FusionNetwork(
            param_dim=param_output_dim,
            coord_dim=coord_output_dim,
            time_dim=time_output_dim,
            hidden_dims=fusion_hidden_dims,
            output_dim=fusion_output_dim,
            activation=activation,
            use_skip=fusion_use_skip,
            dropout=dropout
        )

        # Output heads
        self.field_head = FieldHead(
            input_dim=fusion_output_dim,
            hidden_dim=field_hidden_dim,
            activation=activation
        )

        self.scalar_head = ScalarHead(
            input_dim=fusion_output_dim,
            hidden_dims=scalar_hidden_dims,
            num_scalars=num_scalars,
            activation=activation
        )

        self.num_scalars = num_scalars

    def encode_params(self, params: torch.Tensor) -> torch.Tensor:
        """Encode parameters."""
        return self.param_encoder(params)

    def encode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates."""
        return self.coord_encoder(coords)

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode time."""
        return self.time_encoder(t)

    def forward(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            params: Parameters (batch, 8)
            coords: Coordinates (n_points, 2) or (batch, n_points, 2)
            t: Time (batch,) or (batch, 1)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary with:
            - 'phi': Potential field (batch, n_points)
            - 'scalars': Scalar metrics (batch, num_scalars)
            - Optionally embeddings if return_embeddings=True
        """
        # Encode inputs
        param_emb = self.param_encoder(params)  # (batch, param_dim)
        coord_emb = self.coord_encoder(coords)  # (n_points, coord_dim) or (batch, n_points, coord_dim)
        time_emb = self.time_encoder(t)         # (batch, time_dim)

        # Fuse embeddings
        fused = self.fusion(param_emb, coord_emb, time_emb)  # (batch, n_points, fusion_dim)

        # Predict outputs
        phi = self.field_head(fused).squeeze(-1)  # (batch, n_points)
        scalars = self.scalar_head(fused)         # (batch, num_scalars)

        output = {
            'phi': phi,
            'scalars': scalars
        }

        if return_embeddings:
            output['param_emb'] = param_emb
            output['coord_emb'] = coord_emb
            output['time_emb'] = time_emb
            output['fused'] = fused

        return output

    def predict_field(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict potential field only.

        Args:
            params: Parameters (batch, 8)
            coords: Coordinates (n_points, 2) or (batch, n_points, 2)
            t: Time (batch,) or (batch, 1)

        Returns:
            Potential field (batch, n_points)
        """
        return self.forward(params, coords, t)['phi']

    def predict_scalars(
        self,
        params: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scalar metrics only.

        Args:
            params: Parameters (batch, 8)
            coords: Coordinates (n_points, 2) or (batch, n_points, 2)
            t: Time (batch,) or (batch, 1)

        Returns:
            Scalar metrics (batch, num_scalars)
        """
        return self.forward(params, coords, t)['scalars']

    def forward_with_grad(
        self,
        params: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gradient computation for PDE residual.

        This method keeps coordinate computation graph for autograd.

        Args:
            params: Parameters (batch, 8)
            x: x-coordinates (batch, n_points) with requires_grad=True
            y: y-coordinates (batch, n_points) with requires_grad=True
            t: Time (batch,)

        Returns:
            Dictionary with phi, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dy2
        """
        batch_size = params.shape[0]
        n_points = x.shape[1] if x.dim() > 1 else x.shape[0]

        # Enable grad for coordinates
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        # Stack coordinates
        if x.dim() == 1:
            coords = torch.stack([x, y], dim=-1)  # (n_points, 2)
        else:
            coords = torch.stack([x, y], dim=-1)  # (batch, n_points, 2)

        # Forward pass
        output = self.forward(params, coords, t)
        phi = output['phi']  # (batch, n_points)

        # Compute gradients
        # dphi/dx and dphi/dy
        grad_phi = torch.autograd.grad(
            outputs=phi,
            inputs=[x, y],
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )
        dphi_dx = grad_phi[0]  # (batch, n_points)
        dphi_dy = grad_phi[1]  # (batch, n_points)

        # Second derivatives for Laplacian
        d2phi_dx2 = torch.autograd.grad(
            outputs=dphi_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dphi_dx),
            create_graph=True,
            retain_graph=True
        )[0]

        d2phi_dy2 = torch.autograd.grad(
            outputs=dphi_dy,
            inputs=y,
            grad_outputs=torch.ones_like(dphi_dy),
            create_graph=True,
            retain_graph=True
        )[0]

        return {
            'phi': phi,
            'scalars': output['scalars'],
            'dphi_dx': dphi_dx,
            'dphi_dy': dphi_dy,
            'd2phi_dx2': d2phi_dx2,
            'd2phi_dy2': d2phi_dy2
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config.copy()

    @classmethod
    def from_config(cls, config: dict) -> 'PINN':
        """Create model from configuration dictionary."""
        return cls(**config)

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = 'cpu') -> 'PINN':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


def create_default_pinn(device: str = 'cpu') -> PINN:
    """
    Create PINN with default configuration.

    Returns:
        PINN model with default settings
    """
    model = PINN(
        # Parameter encoder: [8 → 64 → 128 → 256 → 128]
        param_input_dim=8,
        param_hidden_dims=[64, 128, 256],
        param_output_dim=128,
        # Coordinate encoder: Fourier(σ=4.0) + [256 → 128 → 256 → 128]
        coord_input_dim=2,
        num_fourier_features=128,
        fourier_sigma=4.0,
        coord_hidden_dims=[128, 256],
        coord_output_dim=128,
        # Time encoder: PE + [16 → 32 → 64]
        time_num_frequencies=8,
        time_hidden_dims=[32],
        time_output_dim=64,
        # Fusion: [320 → 256 → 256 → 128]
        fusion_hidden_dims=[256, 256],
        fusion_output_dim=128,
        fusion_use_skip=True,
        # Field head: [128 → 64 → 1]
        field_hidden_dim=64,
        # Scalar head: [128 → 128 → 64 → 25]
        scalar_hidden_dims=[128, 64],
        num_scalars=26,
        # General
        activation='gelu',
        dropout=0.0
    )
    return model.to(device)
