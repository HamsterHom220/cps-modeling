"""Loss functions for PINN training."""

from .physics import PDEResidualLoss, BoundaryConditionLoss
from .combined import CombinedLoss

__all__ = ["PDEResidualLoss", "BoundaryConditionLoss", "CombinedLoss"]
