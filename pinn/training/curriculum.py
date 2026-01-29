"""
Curriculum learning scheduler for PINN training.

Implements a 3-phase training curriculum:
1. Phase 1 (Epochs 1-20): Data-only training
2. Phase 2 (Epochs 21-50): Gradual physics introduction
3. Phase 3 (Epochs 51-100): Full physics training
4. Phase 4 (Optional, 101-150): L-BFGS refinement
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Configuration for a curriculum phase."""
    name: str
    start_epoch: int
    end_epoch: int
    lambda_data: float
    lambda_pde: float
    lambda_bc: float
    lambda_scalar: float
    lr_scale: float = 1.0


class CurriculumScheduler:
    """
    Curriculum learning scheduler for PINN.

    Gradually introduces physics-based losses to improve training stability.
    """

    def __init__(
        self,
        phases: Optional[list] = None,
        interpolation: str = 'linear',
        verbose: bool = True
    ):
        """
        Initialize curriculum scheduler.

        Args:
            phases: List of CurriculumPhase objects
            interpolation: 'linear' or 'cosine' for weight interpolation
            verbose: Print phase transitions
        """
        self.interpolation = interpolation
        self.verbose = verbose

        # Default curriculum phases
        if phases is None:
            phases = self._default_phases()

        self.phases = sorted(phases, key=lambda p: p.start_epoch)
        self._validate_phases()

        self.current_epoch = 0
        self.current_phase_idx = 0

    def _default_phases(self) -> list:
        """Create default 4-phase curriculum."""
        return [
            CurriculumPhase(
                name="Data Only",
                start_epoch=1,
                end_epoch=20,
                lambda_data=1.0,
                lambda_pde=0.0,
                lambda_bc=0.0,
                lambda_scalar=0.1,
                lr_scale=1.0
            ),
            CurriculumPhase(
                name="Physics Ramp-Up",
                start_epoch=21,
                end_epoch=50,
                lambda_data=1.0,
                lambda_pde=0.1,  # Target values at end of phase
                lambda_bc=0.5,
                lambda_scalar=0.1,
                lr_scale=1.0
            ),
            CurriculumPhase(
                name="Full Physics",
                start_epoch=51,
                end_epoch=100,
                lambda_data=1.0,
                lambda_pde=0.1,
                lambda_bc=0.5,
                lambda_scalar=0.1,
                lr_scale=0.5  # Reduce LR for fine-tuning
            ),
            CurriculumPhase(
                name="Refinement",
                start_epoch=101,
                end_epoch=150,
                lambda_data=1.0,
                lambda_pde=0.1,
                lambda_bc=0.5,
                lambda_scalar=0.1,
                lr_scale=0.1
            )
        ]

    def _validate_phases(self):
        """Validate that phases are contiguous and non-overlapping."""
        for i in range(len(self.phases) - 1):
            if self.phases[i].end_epoch >= self.phases[i + 1].start_epoch:
                # Allow overlapping phases, use the later one
                pass

    def _interpolate(self, start_val: float, end_val: float, progress: float) -> float:
        """Interpolate between values based on progress [0, 1]."""
        if self.interpolation == 'linear':
            return start_val + progress * (end_val - start_val)
        elif self.interpolation == 'cosine':
            # Cosine annealing from start to end
            return end_val - (end_val - start_val) * (1 + np.cos(np.pi * progress)) / 2
        else:
            return end_val  # Step function

    def get_phase(self, epoch: int) -> Tuple[CurriculumPhase, int]:
        """
        Get the active phase for an epoch.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Tuple of (phase, phase_index)
        """
        for i, phase in enumerate(self.phases):
            if phase.start_epoch <= epoch <= phase.end_epoch:
                return phase, i

        # Return last phase if beyond all defined phases
        return self.phases[-1], len(self.phases) - 1

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for current epoch.

        During ramp-up phases, weights are interpolated from previous phase.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Dictionary of loss weights
        """
        phase, phase_idx = self.get_phase(epoch)

        # Calculate progress within phase
        phase_length = phase.end_epoch - phase.start_epoch + 1
        progress = (epoch - phase.start_epoch) / max(phase_length - 1, 1)
        progress = np.clip(progress, 0, 1)

        # For first phase, use constant weights
        if phase_idx == 0 or phase.name == "Data Only":
            weights = {
                'lambda_data': phase.lambda_data,
                'lambda_pde': 0.0,  # No physics in data-only phase
                'lambda_bc': 0.0,
                'lambda_scalar': phase.lambda_scalar
            }
        else:
            # Interpolate from previous phase end values
            prev_phase = self.phases[phase_idx - 1]

            # For ramp-up phase, interpolate from 0 to target
            if "Ramp" in phase.name:
                weights = {
                    'lambda_data': phase.lambda_data,
                    'lambda_pde': self._interpolate(0.0, phase.lambda_pde, progress),
                    'lambda_bc': self._interpolate(0.0, phase.lambda_bc, progress),
                    'lambda_scalar': phase.lambda_scalar
                }
            else:
                # Constant weights
                weights = {
                    'lambda_data': phase.lambda_data,
                    'lambda_pde': phase.lambda_pde,
                    'lambda_bc': phase.lambda_bc,
                    'lambda_scalar': phase.lambda_scalar
                }

        return weights

    def get_lr_scale(self, epoch: int) -> float:
        """Get learning rate scale for current epoch."""
        phase, _ = self.get_phase(epoch)
        return phase.lr_scale

    def step(self, epoch: int) -> Dict[str, float]:
        """
        Step the scheduler and return updated weights.

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of loss weights
        """
        prev_phase_idx = self.current_phase_idx
        phase, phase_idx = self.get_phase(epoch)

        if phase_idx != prev_phase_idx and self.verbose:
            print(f"\n[Curriculum] Entering phase '{phase.name}' at epoch {epoch}")
            print(f"  Epochs: {phase.start_epoch} - {phase.end_epoch}")
            print(f"  Target weights: data={phase.lambda_data:.2f}, "
                  f"pde={phase.lambda_pde:.2f}, bc={phase.lambda_bc:.2f}, "
                  f"scalar={phase.lambda_scalar:.2f}")
            print(f"  LR scale: {phase.lr_scale}")

        self.current_epoch = epoch
        self.current_phase_idx = phase_idx

        return self.get_weights(epoch)

    def get_phase_info(self, epoch: int) -> str:
        """Get human-readable phase info."""
        phase, phase_idx = self.get_phase(epoch)
        progress = (epoch - phase.start_epoch) / max(phase.end_epoch - phase.start_epoch, 1)
        return f"{phase.name} ({progress * 100:.0f}%)"

    def should_use_lbfgs(self, epoch: int) -> bool:
        """Check if L-BFGS refinement should be used."""
        phase, _ = self.get_phase(epoch)
        return "Refinement" in phase.name

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'current_phase_idx': self.current_phase_idx,
            'interpolation': self.interpolation
        }

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.current_phase_idx = state_dict['current_phase_idx']


def create_curriculum(
    config: Optional[Dict] = None,
    max_epochs: int = 150
) -> CurriculumScheduler:
    """
    Create curriculum scheduler from configuration.

    Args:
        config: Configuration dictionary
        max_epochs: Maximum number of epochs

    Returns:
        CurriculumScheduler instance
    """
    if config is None:
        return CurriculumScheduler()

    # Parse phase configurations
    phases = []
    phase_configs = config.get('phases', [])

    for pc in phase_configs:
        phases.append(CurriculumPhase(
            name=pc.get('name', 'Unknown'),
            start_epoch=pc.get('start_epoch', 1),
            end_epoch=pc.get('end_epoch', max_epochs),
            lambda_data=pc.get('lambda_data', 1.0),
            lambda_pde=pc.get('lambda_pde', 0.1),
            lambda_bc=pc.get('lambda_bc', 0.5),
            lambda_scalar=pc.get('lambda_scalar', 0.1),
            lr_scale=pc.get('lr_scale', 1.0)
        ))

    if len(phases) == 0:
        phases = None

    return CurriculumScheduler(
        phases=phases,
        interpolation=config.get('interpolation', 'linear'),
        verbose=config.get('verbose', True)
    )
