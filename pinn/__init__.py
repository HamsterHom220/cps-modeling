"""
Physics-Informed Neural Network (PINN) for Cathodic Protection System Prediction.

This module implements a PINN that learns to predict cathodic protection system
behavior from FEM-generated data, enforcing physics constraints via PDE residual loss.
"""

from .models import PINN
from .data import CPDataset, Normalizer
from .losses import CombinedLoss
from .training import Trainer, CurriculumScheduler

__version__ = "0.1.0"
__all__ = ["PINN", "CPDataset", "Normalizer", "CombinedLoss", "Trainer", "CurriculumScheduler"]
