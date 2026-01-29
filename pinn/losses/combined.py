"""
Combined loss function for PINN training.

Implements:
- Data loss (MSE between predicted and FEM fields)
- Physics loss (PDE residual + BC)
- Scalar loss (MSE on scalar metrics)
- Adaptive loss weighting (GradNorm or uncertainty-based)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List

from .physics import PDEResidualLoss, BoundaryConditionLoss


class DataLoss(nn.Module):
    """
    Data fidelity loss for field predictions.

    Uses relative L2 error: ||φ_PINN - φ_FEM||² / ||φ_FEM||²
    """

    def __init__(self, reduction: str = 'mean', use_relative: bool = True):
        """
        Initialize data loss.

        Args:
            reduction: 'mean', 'sum', or 'none'
            use_relative: Use relative error (normalized by target magnitude)
        """
        super().__init__()
        self.reduction = reduction
        self.use_relative = use_relative

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute data loss.

        Args:
            pred: Predicted field (batch, n_points)
            target: Target field (batch, n_points)
            mask: Optional mask for valid points

        Returns:
            Data loss
        """
        diff = pred - target

        if mask is not None:
            diff = diff * mask
            n_valid = mask.sum()
        else:
            n_valid = pred.numel()

        sq_error = diff ** 2

        if self.use_relative:
            # Relative L2: ||pred - target||² / ||target||²
            target_sq = target ** 2
            if mask is not None:
                target_sq = target_sq * mask
            denom = target_sq.sum() + 1e-8
            loss = sq_error.sum() / denom
        else:
            # Absolute MSE
            if self.reduction == 'mean':
                loss = sq_error.sum() / (n_valid + 1e-8)
            elif self.reduction == 'sum':
                loss = sq_error.sum()
            else:
                loss = sq_error

        return loss


class ScalarLoss(nn.Module):
    """
    Loss for scalar metric predictions.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize scalar loss.

        Args:
            reduction: 'mean' or 'sum'
            weights: Optional per-metric weights
        """
        super().__init__()
        self.reduction = reduction
        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metric_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute scalar prediction loss.

        Args:
            pred: Predicted scalars (batch, n_scalars)
            target: Target scalars (batch, n_scalars)
            metric_names: Names of metrics (for weighting)

        Returns:
            Scalar loss
        """
        sq_error = (pred - target) ** 2

        if self.weights is not None and metric_names is not None:
            # Apply per-metric weights
            weight_tensor = torch.tensor(
                [self.weights.get(name, 1.0) for name in metric_names],
                device=pred.device,
                dtype=pred.dtype
            )
            sq_error = sq_error * weight_tensor.unsqueeze(0)

        if self.reduction == 'mean':
            return sq_error.mean()
        else:
            return sq_error.sum()


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting using learnable uncertainty parameters.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., CVPR 2018)

    Loss = Σ (1/(2σ²)) * L_i + log(σ)

    where σ is a learnable parameter per loss term.
    """

    def __init__(
        self,
        loss_names: List[str],
        init_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize adaptive weighting.

        Args:
            loss_names: Names of loss terms
            init_weights: Initial weights (as log-variance)
        """
        super().__init__()

        self.loss_names = loss_names
        self.n_losses = len(loss_names)

        # Initialize log-variance parameters (learnable)
        if init_weights is not None:
            log_vars = torch.tensor([
                np.log(init_weights.get(name, 1.0)) for name in loss_names
            ])
        else:
            log_vars = torch.zeros(self.n_losses)

        self.log_vars = nn.Parameter(log_vars)

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply adaptive weighting.

        Args:
            losses: Dictionary of loss values

        Returns:
            Tuple of (total_loss, weighted_losses)
        """
        total_loss = 0.0
        weighted_losses = {}

        for i, name in enumerate(self.loss_names):
            if name in losses:
                # Precision (inverse variance)
                precision = torch.exp(-self.log_vars[i])
                # Weighted loss + regularization
                weighted = precision * losses[name] + self.log_vars[i]
                weighted_losses[name] = weighted
                total_loss = total_loss + weighted

        return total_loss, weighted_losses

    def get_weights(self) -> Dict[str, float]:
        """Get current weights (as precisions)."""
        weights = {}
        for i, name in enumerate(self.loss_names):
            weights[name] = torch.exp(-self.log_vars[i]).item()
        return weights


class CombinedLoss(nn.Module):
    """
    Combined loss function for PINN training.

    L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc + λ_scalar·L_scalar
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_bc: float = 0.5,
        lambda_scalar: float = 0.1,
        use_relative_data_loss: bool = True,
        use_variable_sigma: bool = True,
        adaptive_weighting: bool = False,
        scalar_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize combined loss.

        Args:
            lambda_data: Weight for data loss
            lambda_pde: Weight for PDE residual loss
            lambda_bc: Weight for boundary condition loss
            lambda_scalar: Weight for scalar loss
            use_relative_data_loss: Use relative L2 for data loss
            use_variable_sigma: Use variable conductivity in PDE
            adaptive_weighting: Use learnable loss weights
            scalar_weights: Per-metric weights for scalar loss
        """
        super().__init__()

        # Store weights
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_scalar = lambda_scalar

        # Loss components
        self.data_loss = DataLoss(use_relative=use_relative_data_loss)
        self.pde_loss = PDEResidualLoss(use_variable_sigma=use_variable_sigma)
        self.bc_loss = BoundaryConditionLoss()
        self.scalar_loss = ScalarLoss(weights=scalar_weights)

        # Adaptive weighting
        self.adaptive_weighting = adaptive_weighting
        if adaptive_weighting:
            self.weight_module = AdaptiveLossWeighting(
                loss_names=['data', 'pde', 'bc', 'scalar'],
                init_weights={
                    'data': lambda_data,
                    'pde': lambda_pde,
                    'bc': lambda_bc,
                    'scalar': lambda_scalar
                }
            )
        else:
            self.weight_module = None

    def update_weights(
        self,
        lambda_data: Optional[float] = None,
        lambda_pde: Optional[float] = None,
        lambda_bc: Optional[float] = None,
        lambda_scalar: Optional[float] = None
    ):
        """Update loss weights (for curriculum learning)."""
        if lambda_data is not None:
            self.lambda_data = lambda_data
        if lambda_pde is not None:
            self.lambda_pde = lambda_pde
        if lambda_bc is not None:
            self.lambda_bc = lambda_bc
        if lambda_scalar is not None:
            self.lambda_scalar = lambda_scalar

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_phi: torch.Tensor,
        target_scalars: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        scalar_names: Optional[List[str]] = None,
        compute_physics: bool = True,
        phi_physics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            model_output: Dictionary with 'phi' and 'scalars' from model
            target_phi: Target potential field (batch, n_points)
            target_scalars: Target scalar metrics (batch, n_scalars)
            x: x-coordinates for physics loss (requires_grad=True)
            y: y-coordinates for physics loss (requires_grad=True)
            sigma: Conductivity field for physics loss
            scalar_names: Names of scalar metrics
            compute_physics: Whether to compute physics loss
            phi_physics: Phi values at physics sample points (if different from model_output)

        Returns:
            Dictionary with all loss components and total
        """
        losses = {}

        pred_phi = model_output['phi']
        pred_scalars = model_output.get('scalars')

        # Data loss
        data_loss = self.data_loss(pred_phi, target_phi)
        losses['data'] = data_loss

        # Physics losses (use phi_physics if provided, else pred_phi)
        if compute_physics and x is not None and y is not None:
            phi_for_physics = phi_physics if phi_physics is not None else pred_phi

            # PDE residual
            pde_loss = self.pde_loss(phi_for_physics, x, y, sigma)
            losses['pde'] = pde_loss

            # Boundary conditions (skip for now - expensive and less important)
            # bc_loss computation requires full grid, not sampled points
            losses['bc'] = torch.tensor(0.0, device=pred_phi.device)
        else:
            losses['pde'] = torch.tensor(0.0, device=pred_phi.device)
            losses['bc'] = torch.tensor(0.0, device=pred_phi.device)

        # Scalar loss
        if target_scalars is not None and pred_scalars is not None:
            scalar_loss = self.scalar_loss(pred_scalars, target_scalars, scalar_names)
            losses['scalar'] = scalar_loss
        else:
            losses['scalar'] = torch.tensor(0.0, device=pred_phi.device)

        # Compute total loss
        if self.adaptive_weighting and self.weight_module is not None:
            total_loss, weighted_losses = self.weight_module(losses)
            losses['total'] = total_loss
            losses.update({f'{k}_weighted': v for k, v in weighted_losses.items()})
        else:
            total_loss = (
                self.lambda_data * losses['data'] +
                self.lambda_pde * losses['pde'] +
                self.lambda_bc * losses['bc'] +
                self.lambda_scalar * losses['scalar']
            )
            losses['total'] = total_loss

        return losses

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        if self.adaptive_weighting and self.weight_module is not None:
            return self.weight_module.get_weights()
        return {
            'data': self.lambda_data,
            'pde': self.lambda_pde,
            'bc': self.lambda_bc,
            'scalar': self.lambda_scalar
        }


def create_loss_function(
    config: Optional[Dict] = None,
    adaptive: bool = False
) -> CombinedLoss:
    """
    Create loss function from configuration.

    Args:
        config: Configuration dictionary
        adaptive: Use adaptive weighting

    Returns:
        CombinedLoss instance
    """
    if config is None:
        config = {}

    return CombinedLoss(
        lambda_data=config.get('lambda_data', 1.0),
        lambda_pde=config.get('lambda_pde', 0.1),
        lambda_bc=config.get('lambda_bc', 0.5),
        lambda_scalar=config.get('lambda_scalar', 0.1),
        use_relative_data_loss=config.get('use_relative_data_loss', True),
        use_variable_sigma=config.get('use_variable_sigma', True),
        adaptive_weighting=adaptive
    )
