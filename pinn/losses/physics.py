"""
Physics-based losses for PINN training.

Implements:
- PDE Residual Loss: ∇·(σ∇φ) = 0 (Laplace equation with variable conductivity)
- Boundary Condition Loss: Robin BC at pipe/anode interfaces
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class PhysicsConstants:
    """Physical constants for cathodic protection system."""

    # Electrochemical constants
    FARADAY = 96485.0           # C/mol
    GAS_CONSTANT = 8.314        # J/(mol·K)

    # Equilibrium potentials (vs Cu/CuSO4 reference)
    E_STEEL_EQ = -0.65          # V - steel equilibrium potential
    E_ANODE_EQ = 1.2            # V - ICCP anode potential
    E_PROTECTION = -0.85        # V - cathodic protection criterion

    # Exchange current densities
    I0_STEEL = 1e-6             # A/m²
    I0_ANODE = 1e-4             # A/m²

    # Domain geometry (in meters)
    DOMAIN_WIDTH = 20.0         # m
    DOMAIN_HEIGHT = 8.0         # m

    # Pipe geometry
    PIPE_CENTER_Y = 4.0         # m
    PIPE_X_MIN = 5.0            # m
    PIPE_X_MAX = 15.0           # m
    PIPE_RADIUS = 0.2           # m

    # Anode geometry
    ANODE_CENTER_Y = 1.5        # m
    ANODE_X_MIN = 9.0           # m
    ANODE_X_MAX = 11.0          # m

    # Grid resolution
    NX = 40
    NY = 20


class PDEResidualLoss(nn.Module):
    """
    PDE Residual Loss for Laplace equation with variable conductivity.

    Governing equation: ∇·(σ∇φ) = 0
    Expanded: σ(∂²φ/∂x² + ∂²φ/∂y²) + ∇σ·∇φ = 0

    For uniform conductivity: ∂²φ/∂x² + ∂²φ/∂y² = 0 (standard Laplace)
    """

    def __init__(
        self,
        use_variable_sigma: bool = True,
        normalize_residual: bool = True
    ):
        """
        Initialize PDE residual loss.

        Args:
            use_variable_sigma: Include conductivity gradient terms
            normalize_residual: Normalize by domain size
        """
        super().__init__()
        self.use_variable_sigma = use_variable_sigma
        self.normalize_residual = normalize_residual

    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute PDE residual loss.

        Args:
            phi: Potential field (batch, n_points) - computed with grad enabled
            x: x-coordinates (batch, n_points) with requires_grad=True
            y: y-coordinates (batch, n_points) with requires_grad=True
            sigma: Conductivity field (batch, n_points), optional

        Returns:
            Mean squared PDE residual
        """
        # First derivatives
        dphi_dx = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        dphi_dy = torch.autograd.grad(
            phi, y,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        # Second derivatives (Laplacian components)
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx, x,
            grad_outputs=torch.ones_like(dphi_dx),
            create_graph=True,
            retain_graph=True
        )[0]

        d2phi_dy2 = torch.autograd.grad(
            dphi_dy, y,
            grad_outputs=torch.ones_like(dphi_dy),
            create_graph=True,
            retain_graph=True
        )[0]

        if self.use_variable_sigma and sigma is not None:
            # Weighted Laplace: σ∇²φ = 0
            # Note: We ignore the ∇σ·∇φ term because sigma is data (not computed),
            # so we can't compute its gradients via autograd. This is a good
            # approximation when sigma varies slowly.
            residual = sigma * (d2phi_dx2 + d2phi_dy2)
        else:
            # Standard Laplace: ∇²φ = 0
            residual = d2phi_dx2 + d2phi_dy2

        # Normalize if requested
        if self.normalize_residual:
            # Scale by domain size (heuristic normalization)
            scale = 1.0 / (PhysicsConstants.DOMAIN_WIDTH * PhysicsConstants.DOMAIN_HEIGHT)
            residual = residual * scale

        # Mean squared residual
        loss = torch.mean(residual ** 2)

        return loss

    def compute_residual(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute PDE residual (not squared) for analysis.

        Returns:
            Residual tensor (batch, n_points)
        """
        dphi_dx = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=True
        )[0]

        dphi_dy = torch.autograd.grad(
            phi, y,
            grad_outputs=torch.ones_like(phi),
            create_graph=False,
            retain_graph=True
        )[0]

        d2phi_dx2 = torch.autograd.grad(
            dphi_dx, x,
            grad_outputs=torch.ones_like(dphi_dx),
            create_graph=False,
            retain_graph=True
        )[0]

        d2phi_dy2 = torch.autograd.grad(
            dphi_dy, y,
            grad_outputs=torch.ones_like(dphi_dy),
            create_graph=False,
            retain_graph=True
        )[0]

        if self.use_variable_sigma and sigma is not None:
            # Weighted Laplace: σ∇²φ = 0
            residual = sigma * (d2phi_dx2 + d2phi_dy2)
        else:
            residual = d2phi_dx2 + d2phi_dy2

        return residual


class BoundaryConditionLoss(nn.Module):
    """
    Boundary condition loss for cathodic protection system.

    Implements Robin (mixed) boundary conditions:
    - At pipe: σ * ∂φ/∂n = (φ - E_pipe) / R_pipe
    - At anode: σ * ∂φ/∂n = (φ - E_anode) / R_anode
    - At domain boundaries: ∂φ/∂n = 0 (natural BC)

    Note: R is the linearized polarization resistance.
    """

    def __init__(
        self,
        e_pipe: float = PhysicsConstants.E_STEEL_EQ,
        e_anode: float = PhysicsConstants.E_ANODE_EQ,
        default_r_pipe: float = 10.0,
        default_r_anode: float = 1.0
    ):
        """
        Initialize BC loss.

        Args:
            e_pipe: Pipe equilibrium potential
            e_anode: Anode equilibrium potential
            default_r_pipe: Default pipe polarization resistance
            default_r_anode: Default anode polarization resistance
        """
        super().__init__()
        self.e_pipe = e_pipe
        self.e_anode = e_anode
        self.default_r_pipe = default_r_pipe
        self.default_r_anode = default_r_anode

    def _get_boundary_masks(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        tol: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Create masks for different boundary regions.

        Args:
            x: x-coordinates (batch, n_points) - denormalized to [0, 20]
            y: y-coordinates (batch, n_points) - denormalized to [0, 8]
            tol: Tolerance for boundary detection

        Returns:
            Dictionary of boolean masks
        """
        # Pipe boundary (horizontal pipe at y=4, x in [5, 15])
        pipe_mask = (
            (torch.abs(y - PhysicsConstants.PIPE_CENTER_Y) < tol) &
            (x >= PhysicsConstants.PIPE_X_MIN) &
            (x <= PhysicsConstants.PIPE_X_MAX)
        )

        # Anode boundary (y=1.5, x in [9, 11])
        anode_mask = (
            (torch.abs(y - PhysicsConstants.ANODE_CENTER_Y) < tol) &
            (x >= PhysicsConstants.ANODE_X_MIN) &
            (x <= PhysicsConstants.ANODE_X_MAX)
        )

        # Domain boundaries (natural BC)
        left_mask = x < tol
        right_mask = x > (PhysicsConstants.DOMAIN_WIDTH - tol)
        bottom_mask = y < tol
        top_mask = y > (PhysicsConstants.DOMAIN_HEIGHT - tol)

        return {
            'pipe': pipe_mask,
            'anode': anode_mask,
            'left': left_mask,
            'right': right_mask,
            'bottom': bottom_mask,
            'top': top_mask
        }

    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: torch.Tensor,
        r_pipe: Optional[torch.Tensor] = None,
        r_anode: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            phi: Potential field (batch, n_points)
            x: x-coordinates (batch, n_points) with requires_grad=True
            y: y-coordinates (batch, n_points) with requires_grad=True
            sigma: Conductivity (batch, n_points)
            r_pipe: Pipe resistance (batch,), optional
            r_anode: Anode resistance (batch,), optional

        Returns:
            Total BC loss
        """
        if r_pipe is None:
            r_pipe = torch.full((phi.shape[0],), self.default_r_pipe,
                               device=phi.device, dtype=phi.dtype)
        if r_anode is None:
            r_anode = torch.full((phi.shape[0],), self.default_r_anode,
                                device=phi.device, dtype=phi.dtype)

        # Get boundary masks
        masks = self._get_boundary_masks(x, y)

        # Compute gradients
        dphi_dx = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        dphi_dy = torch.autograd.grad(
            phi, y,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        total_loss = 0.0
        n_terms = 0

        # Pipe BC: σ * ∂φ/∂n = (φ - E_pipe) / R_pipe
        # For pipe at y=4, normal is approximately ±y direction
        if masks['pipe'].any():
            pipe_flux = sigma * dphi_dy  # Normal flux
            # Expand r_pipe for broadcasting
            r_pipe_expanded = r_pipe.unsqueeze(-1).expand_as(phi)
            pipe_robin = (phi - self.e_pipe) / r_pipe_expanded
            pipe_residual = pipe_flux - pipe_robin
            pipe_loss = torch.mean(pipe_residual[masks['pipe']] ** 2)
            total_loss = total_loss + pipe_loss
            n_terms += 1

        # Anode BC: σ * ∂φ/∂n = (φ - E_anode) / R_anode
        if masks['anode'].any():
            anode_flux = sigma * dphi_dy
            r_anode_expanded = r_anode.unsqueeze(-1).expand_as(phi)
            anode_robin = (phi - self.e_anode) / r_anode_expanded
            anode_residual = anode_flux - anode_robin
            anode_loss = torch.mean(anode_residual[masks['anode']] ** 2)
            total_loss = total_loss + anode_loss
            n_terms += 1

        # Natural BC on domain boundaries: ∂φ/∂n = 0
        # Left/right: dphi_dx = 0
        for name in ['left', 'right']:
            if masks[name].any():
                bc_residual = dphi_dx[masks[name]]
                total_loss = total_loss + torch.mean(bc_residual ** 2)
                n_terms += 1

        # Top/bottom: dphi_dy = 0
        for name in ['bottom', 'top']:
            if masks[name].any():
                bc_residual = dphi_dy[masks[name]]
                total_loss = total_loss + torch.mean(bc_residual ** 2)
                n_terms += 1

        # Average over BC terms
        if n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss


class CurrentConservationLoss(nn.Module):
    """
    Current conservation loss.

    Ensures that the total current from the anode equals
    the total current to the pipe (minus any losses).
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize current conservation loss.

        Args:
            tolerance: Acceptable relative error
        """
        super().__init__()
        self.tolerance = tolerance

    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute current conservation loss.

        Args:
            phi: Potential field (batch, n_points)
            x: x-coordinates
            y: y-coordinates
            sigma: Conductivity

        Returns:
            Current conservation loss
        """
        # Get boundary masks
        bc_loss = BoundaryConditionLoss()
        masks = bc_loss._get_boundary_masks(x, y)

        # Compute normal flux at pipe and anode
        dphi_dy = torch.autograd.grad(
            phi, y,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]

        flux = sigma * dphi_dy

        # Total current at pipe (cathode)
        if masks['pipe'].any():
            pipe_current = flux[masks['pipe']].sum()
        else:
            pipe_current = torch.tensor(0.0, device=phi.device)

        # Total current at anode
        if masks['anode'].any():
            anode_current = flux[masks['anode']].sum()
        else:
            anode_current = torch.tensor(0.0, device=phi.device)

        # Conservation: I_anode + I_pipe ≈ 0 (opposite signs)
        current_balance = anode_current + pipe_current

        # Relative error (normalized by anode current magnitude)
        denom = torch.abs(anode_current) + 1e-8
        conservation_error = (current_balance / denom) ** 2

        return conservation_error


def compute_physics_loss(
    model,
    params: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    pde_weight: float = 1.0,
    bc_weight: float = 1.0,
    use_variable_sigma: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute all physics losses.

    Args:
        model: PINN model
        params: Input parameters
        x: x-coordinates (requires_grad=True)
        y: y-coordinates (requires_grad=True)
        t: Time
        sigma: Conductivity field
        pde_weight: Weight for PDE loss
        bc_weight: Weight for BC loss
        use_variable_sigma: Use variable conductivity

    Returns:
        Dictionary with individual and total losses
    """
    # Ensure gradients are enabled
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    # Stack coordinates
    coords = torch.stack([x, y], dim=-1)

    # Forward pass
    output = model(params, coords, t)
    phi = output['phi']

    # PDE loss
    pde_loss_fn = PDEResidualLoss(use_variable_sigma=use_variable_sigma)
    pde_loss = pde_loss_fn(phi, x, y, sigma)

    # BC loss
    bc_loss_fn = BoundaryConditionLoss()
    bc_loss = bc_loss_fn(phi, x, y, sigma if sigma is not None else torch.ones_like(phi))

    # Total physics loss
    total_physics = pde_weight * pde_loss + bc_weight * bc_loss

    return {
        'pde_loss': pde_loss,
        'bc_loss': bc_loss,
        'physics_loss': total_physics
    }
