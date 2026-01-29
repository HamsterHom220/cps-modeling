"""
Training loop for PINN.

Implements:
- Full training loop with validation
- Curriculum learning integration
- Checkpointing and logging
- Mixed precision training
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ..models import PINN
from ..losses import CombinedLoss
from ..data import Normalizer
from .curriculum import CurriculumScheduler


class Trainer:
    """
    PINN Trainer with curriculum learning support.
    """

    def __init__(
        self,
        model: PINN,
        loss_fn: CombinedLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: Normalizer,
        curriculum: Optional[CurriculumScheduler] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        max_epochs: int = 150,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: str = './experiments',
        log_interval: int = 10,
        val_interval: int = 1,
        save_interval: int = 10,
        early_stopping_patience: int = 20,
        compute_physics_loss: bool = True,
        physics_sample_size: int = 100
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model
            loss_fn: Combined loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            normalizer: Data normalizer
            curriculum: Curriculum scheduler
            optimizer: Optimizer (AdamW if None)
            scheduler: LR scheduler (cosine annealing if None)
            device: Training device
            max_epochs: Maximum epochs
            grad_clip: Gradient clipping max norm
            use_amp: Use automatic mixed precision
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging interval (batches)
            val_interval: Validation interval (epochs)
            save_interval: Checkpoint interval (epochs)
            early_stopping_patience: Epochs without improvement before stopping
            compute_physics_loss: Whether to compute physics losses
            physics_sample_size: Number of points to sample for physics loss (default 100)
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Keep normalizer on CPU (shared with DataLoader workers)
        self.normalizer = normalizer
        self.curriculum = curriculum if curriculum is not None else CurriculumScheduler()
        self.device = device
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device == 'cuda'
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience
        self.compute_physics_loss = compute_physics_loss
        self.physics_sample_size = physics_sample_size

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduler
        if scheduler is None:
            # Warmup + cosine annealing
            warmup_epochs = min(5, max_epochs // 3)  # Cap warmup to avoid pct_start > 1
            pct_start = warmup_epochs / max_epochs if max_epochs > 0 else 0.1
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=1e-3,
                epochs=max_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=max(0.01, min(pct_start, 0.3)),  # Clamp to valid range
                anneal_strategy='cos'
            )
        else:
            self.scheduler = scheduler

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch_time': [],
            'loss_weights': []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch (1-indexed)

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'data': 0.0,
            'pde': 0.0,
            'bc': 0.0,
            'scalar': 0.0
        }
        n_batches = 0

        # Get curriculum weights
        weights = self.curriculum.step(epoch)
        self.loss_fn.update_weights(**weights)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            # Get coordinates for physics loss
            x_grid = batch['x_grid']  # (batch, ny, nx) or similar
            y_grid = batch['y_grid']

            # Flatten spatial dimensions if needed
            if x_grid.dim() == 3:
                batch_size = x_grid.shape[0]
                x_flat = x_grid.reshape(batch_size, -1)
                y_flat = y_grid.reshape(batch_size, -1)
            else:
                x_flat = x_grid.flatten().unsqueeze(0).expand(batch['params'].shape[0], -1)
                y_flat = y_grid.flatten().unsqueeze(0).expand(batch['params'].shape[0], -1)

            # Stack coordinates for data loss (all points)
            coords = torch.stack([x_flat, y_flat], dim=-1)

            # Forward pass
            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                # Model forward for data loss (all points)
                output = self.model(batch['params'], coords, batch['time'])

                # Flatten target phi if needed
                target_phi = batch['phi']
                if target_phi.dim() == 3:
                    target_phi = target_phi.reshape(target_phi.shape[0], -1)

                # Get sigma if available
                sigma = batch.get('sigma')
                if sigma is not None and sigma.dim() == 3:
                    sigma = sigma.reshape(sigma.shape[0], -1)

                # For physics loss, sample subset of points and do separate forward pass
                # IMPORTANT: Physics loss with autograd second derivatives must use float32
                # to avoid numerical instability. We exit autocast for this computation.
                phi_physics, x_physics, y_physics, sigma_physics = None, None, None, None

            # Physics computation must be in float32 (outside autocast)
            if self.compute_physics_loss:
                n_points = x_flat.shape[1]
                n_physics = min(self.physics_sample_size, n_points)
                idx = torch.randperm(n_points, device=self.device)[:n_physics]

                # Sample coordinates with gradients enabled (in float32)
                x_physics = x_flat[:, idx].detach().float().requires_grad_(True)
                y_physics = y_flat[:, idx].detach().float().requires_grad_(True)

                # Forward pass at sampled points in float32 (needed for autograd)
                coords_physics = torch.stack([x_physics, y_physics], dim=-1)
                with torch.cuda.amp.autocast(enabled=False):
                    output_physics = self.model(batch['params'].float(), coords_physics, batch['time'].float())
                phi_physics = output_physics['phi']

                if sigma is not None:
                    sigma_physics = sigma[:, idx].float()

            # Now compute loss (data loss can still use AMP, physics uses float32)
            with autocast('cuda', enabled=self.use_amp):

                # Compute loss
                losses = self.loss_fn(
                    model_output=output,
                    target_phi=target_phi,
                    target_scalars=batch.get('scalars'),
                    x=x_physics,
                    y=y_physics,
                    sigma=sigma_physics,
                    phi_physics=phi_physics,
                    compute_physics=self.compute_physics_loss
                )

            loss = losses['total']

            # Skip batch if loss is NaN/Inf (defensive measure)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...')
                self.optimizer.zero_grad()
                continue

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # LR scheduler step (per-batch for OneCycleLR)
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'data': f'{losses["data"].item():.4f}',
                'pde': f'{losses["pde"].item():.4f}'
            })

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(n_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch

        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'data': 0.0,
            'pde': 0.0,
            'bc': 0.0,
            'scalar': 0.0
        }
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            # Get coordinates
            x_grid = batch['x_grid']
            y_grid = batch['y_grid']

            if x_grid.dim() == 3:
                batch_size = x_grid.shape[0]
                x_flat = x_grid.reshape(batch_size, -1)
                y_flat = y_grid.reshape(batch_size, -1)
            else:
                x_flat = x_grid.flatten().unsqueeze(0).expand(batch['params'].shape[0], -1)
                y_flat = y_grid.flatten().unsqueeze(0).expand(batch['params'].shape[0], -1)

            coords = torch.stack([x_flat, y_flat], dim=-1)

            # Forward pass
            output = self.model(batch['params'], coords, batch['time'])

            # Flatten target
            target_phi = batch['phi']
            if target_phi.dim() == 3:
                target_phi = target_phi.reshape(target_phi.shape[0], -1)

            sigma = batch.get('sigma')
            if sigma is not None and sigma.dim() == 3:
                sigma = sigma.reshape(sigma.shape[0], -1)

            # Compute loss (without physics for faster validation)
            losses = self.loss_fn(
                model_output=output,
                target_phi=target_phi,
                target_scalars=batch.get('scalars'),
                x=None,
                y=None,
                sigma=sigma,
                compute_physics=False
            )

            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            n_batches += 1

        for key in val_losses:
            val_losses[key] /= max(n_batches, 1)

        return val_losses

    def train(self) -> Dict[str, List]:
        """
        Full training loop.

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting PINN Training")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            if epoch % self.val_interval == 0:
                val_losses = self.validate(epoch)
            else:
                val_losses = {'total': float('inf'), 'data': 0, 'pde': 0, 'bc': 0, 'scalar': 0}

            epoch_time = time.time() - epoch_start

            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            print(f"Epoch {epoch:3d}/{self.max_epochs} | "
                  f"Train: {train_losses['total']:.4f} | "
                  f"Val: {val_losses['total']:.4f} | "
                  f"Data: {train_losses['data']:.4f} | "
                  f"PDE: {train_losses['pde']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Phase: {self.curriculum.get_phase_info(epoch)}")

            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            self.history['loss_weights'].append(self.loss_fn.get_weights())

            # Check for improvement
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best.pt')
            else:
                self.epochs_without_improvement += 1

            # Regular checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch:03d}.pt')

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.early_stopping_patience} epochs)")
                break

            # LR scheduler step (per-epoch for non-OneCycleLR)
            if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

        # Save final checkpoint and history
        self.save_checkpoint('final.pt')
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'curriculum_state_dict': self.curriculum.state_dict(),
            'best_val_loss': self.best_val_loss,
            'normalizer_state_dict': self.normalizer.state_dict(),
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.curriculum.load_state_dict(checkpoint['curriculum_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'history.json'

        # Convert numpy types for JSON serialization
        serializable_history = {}
        for key, values in self.history.items():
            if key == 'loss_weights':
                serializable_history[key] = values
            else:
                serializable_history[key] = [float(v) if isinstance(v, (np.floating, float)) else v
                                              for v in values]

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)


def create_trainer(
    model: PINN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: Normalizer,
    config: Optional[Dict] = None,
    device: str = 'cuda'
) -> Trainer:
    """
    Create trainer from configuration.

    Args:
        model: PINN model
        train_loader: Training data loader
        val_loader: Validation data loader
        normalizer: Data normalizer
        config: Configuration dictionary
        device: Training device

    Returns:
        Trainer instance
    """
    if config is None:
        config = {}

    # Create loss function
    loss_fn = CombinedLoss(
        lambda_data=config.get('lambda_data', 1.0),
        lambda_pde=config.get('lambda_pde', 0.1),
        lambda_bc=config.get('lambda_bc', 0.5),
        lambda_scalar=config.get('lambda_scalar', 0.1),
        use_relative_data_loss=config.get('use_relative_data_loss', True),
        use_variable_sigma=config.get('use_variable_sigma', True),
        adaptive_weighting=config.get('adaptive_weighting', False)
    )

    # Create curriculum
    curriculum = CurriculumScheduler(
        interpolation=config.get('curriculum_interpolation', 'linear'),
        verbose=config.get('verbose', True)
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        betas=config.get('betas', (0.9, 0.999))
    )

    return Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        curriculum=curriculum,
        optimizer=optimizer,
        device=device,
        max_epochs=config.get('max_epochs', 150),
        grad_clip=config.get('grad_clip', 1.0),
        use_amp=config.get('use_amp', True),
        checkpoint_dir=config.get('checkpoint_dir', './experiments'),
        log_interval=config.get('log_interval', 10),
        val_interval=config.get('val_interval', 1),
        save_interval=config.get('save_interval', 10),
        early_stopping_patience=config.get('early_stopping_patience', 20),
        compute_physics_loss=config.get('compute_physics_loss', True)
    )
