"""
Encoder networks for PINN.

Implements:
- ParameterEncoder: Encodes 8D input parameters
- CoordinateEncoder: Encodes spatial coordinates with Fourier features
- TimeEncoder: Encodes time with positional encoding
- FusionNetwork: Combines encoded representations
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .fourier import FourierFeatures, PositionalEncoding


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation and normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'gelu',
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ('gelu', 'relu', 'tanh', 'silu')
            output_activation: Activation for output layer (None for linear)
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation functions
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'softplus': nn.Softplus(),
            None: nn.Identity()
        }
        self.act = activations[activation]
        self.output_act = activations[output_activation]

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)

            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if not is_last:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(self.act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(self.output_act)

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParameterEncoder(nn.Module):
    """
    Encodes 8D CPS parameters into a latent representation.

    Architecture: MLP [8 → 64 → 128 → 256 → 128]
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 128,
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        """
        Initialize parameter encoder.

        Args:
            input_dim: Number of input parameters (default: 8)
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_activation=None,
            use_layer_norm=True,
            dropout=dropout
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Encode parameters.

        Args:
            params: Parameter tensor of shape (batch, 8)

        Returns:
            Encoded parameters of shape (batch, output_dim)
        """
        return self.encoder(params)


class CoordinateEncoder(nn.Module):
    """
    Encodes spatial coordinates (x, y) with Fourier features.

    Architecture: Fourier(σ=4.0) + MLP [256 → 128 → 256 → 128]
    Uses Tanh activation for smooth coordinate embeddings.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_fourier_features: int = 128,
        fourier_sigma: float = 4.0,
        hidden_dims: List[int] = [128, 256],
        output_dim: int = 128,
        activation: str = 'tanh'
    ):
        """
        Initialize coordinate encoder.

        Args:
            input_dim: Coordinate dimension (default: 2 for x, y)
            num_fourier_features: Number of Fourier frequency components
            fourier_sigma: Std of Fourier frequency distribution
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            activation: Activation function
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Fourier feature encoding
        self.fourier = FourierFeatures(
            input_dim=input_dim,
            num_frequencies=num_fourier_features,
            sigma=fourier_sigma,
            include_input=True
        )

        # MLP on Fourier features
        self.mlp = MLP(
            input_dim=self.fourier.output_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_activation=None,
            use_layer_norm=False
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates.

        Args:
            coords: Coordinate tensor of shape (..., 2)

        Returns:
            Encoded coordinates of shape (..., output_dim)
        """
        fourier_features = self.fourier(coords)
        return self.mlp(fourier_features)


class TimeEncoder(nn.Module):
    """
    Encodes time with positional encoding.

    Architecture: PositionalEncoding + MLP [16 → 32 → 64]
    """

    def __init__(
        self,
        num_frequencies: int = 8,
        hidden_dims: List[int] = [32],
        output_dim: int = 64,
        activation: str = 'gelu'
    ):
        """
        Initialize time encoder.

        Args:
            num_frequencies: Number of positional encoding frequencies
            hidden_dims: Hidden layer dimensions
            output_dim: Output embedding dimension
            activation: Activation function
        """
        super().__init__()

        self.output_dim = output_dim

        # Positional encoding for time
        self.pos_enc = PositionalEncoding(
            input_dim=1,
            num_frequencies=num_frequencies,
            include_input=True
        )

        # MLP on positional features
        self.mlp = MLP(
            input_dim=self.pos_enc.output_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_activation=None
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time.

        Args:
            t: Time tensor of shape (batch,) or (batch, 1)

        Returns:
            Encoded time of shape (batch, output_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        pos_features = self.pos_enc(t)
        return self.mlp(pos_features)


class FusionNetwork(nn.Module):
    """
    Fuses parameter, coordinate, and time embeddings.

    Uses skip connections for better gradient flow.
    Architecture: MLP [320 → 256 → 256 → 128] with skip connections
    """

    def __init__(
        self,
        param_dim: int = 128,
        coord_dim: int = 128,
        time_dim: int = 64,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 128,
        activation: str = 'gelu',
        use_skip: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize fusion network.

        Args:
            param_dim: Parameter embedding dimension
            coord_dim: Coordinate embedding dimension
            time_dim: Time embedding dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            use_skip: Whether to use skip connections
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = param_dim + coord_dim + time_dim
        self.output_dim = output_dim
        self.use_skip = use_skip

        # Activation
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'silu': nn.SiLU()
        }
        self.act = activations[activation]

        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [self.input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No norm on output
                self.norms.append(nn.LayerNorm(dims[i + 1]))

        # Skip projection if dimensions don't match
        if use_skip and self.input_dim != output_dim:
            self.skip_proj = nn.Linear(self.input_dim, output_dim)
        else:
            self.skip_proj = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        param_emb: torch.Tensor,
        coord_emb: torch.Tensor,
        time_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse embeddings.

        Args:
            param_emb: Parameter embedding (batch, param_dim) or (batch, n_points, param_dim)
            coord_emb: Coordinate embedding (batch, n_points, coord_dim) or (n_points, coord_dim)
            time_emb: Time embedding (batch, time_dim)

        Returns:
            Fused representation (batch, n_points, output_dim)
        """
        # Handle broadcasting for different input shapes

        # If coord_emb is 2D (n_points, coord_dim), expand for batch
        if coord_emb.dim() == 2:
            # coord_emb: (n_points, coord_dim)
            # param_emb: (batch, param_dim)
            # time_emb: (batch, time_dim)
            batch_size = param_emb.shape[0]
            n_points = coord_emb.shape[0]

            # Expand param_emb: (batch, 1, param_dim) -> (batch, n_points, param_dim)
            param_emb = param_emb.unsqueeze(1).expand(-1, n_points, -1)

            # Expand time_emb: (batch, 1, time_dim) -> (batch, n_points, time_dim)
            time_emb = time_emb.unsqueeze(1).expand(-1, n_points, -1)

            # Expand coord_emb: (1, n_points, coord_dim) -> (batch, n_points, coord_dim)
            coord_emb = coord_emb.unsqueeze(0).expand(batch_size, -1, -1)

        elif coord_emb.dim() == 3:
            # coord_emb: (batch, n_points, coord_dim)
            n_points = coord_emb.shape[1]

            if param_emb.dim() == 2:
                param_emb = param_emb.unsqueeze(1).expand(-1, n_points, -1)

            if time_emb.dim() == 2:
                time_emb = time_emb.unsqueeze(1).expand(-1, n_points, -1)

        # Concatenate
        x = torch.cat([param_emb, coord_emb, time_emb], dim=-1)
        x_input = x

        # Forward through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)

        # Skip connection
        if self.use_skip:
            if self.skip_proj is not None:
                x = x + self.skip_proj(x_input)
            else:
                x = x + x_input

        return x


class FieldHead(nn.Module):
    """
    Output head for field prediction (potential φ at each point).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = 'gelu'
    ):
        """
        Initialize field head.

        Args:
            input_dim: Input dimension from fusion network
            hidden_dim: Hidden layer dimension
            activation: Activation function
        """
        super().__init__()

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            output_dim=1,
            activation=activation,
            output_activation=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict potential field.

        Args:
            x: Fused representation (batch, n_points, input_dim)

        Returns:
            Potential predictions (batch, n_points, 1)
        """
        return self.net(x)


class ScalarHead(nn.Module):
    """
    Output head for scalar metric predictions.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [128, 64],
        num_scalars: int = 25,
        activation: str = 'gelu'
    ):
        """
        Initialize scalar head.

        Args:
            input_dim: Input dimension from fusion network
            hidden_dims: Hidden layer dimensions
            num_scalars: Number of scalar outputs
            activation: Activation function
        """
        super().__init__()

        self.num_scalars = num_scalars

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_scalars,
            activation=activation,
            output_activation=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar metrics.

        Args:
            x: Fused representation. If 3D (batch, n_points, dim), will pool first.

        Returns:
            Scalar predictions (batch, num_scalars)
        """
        # If input has spatial dimension, pool over points
        if x.dim() == 3:
            x = x.mean(dim=1)  # Global average pooling

        return self.net(x)
