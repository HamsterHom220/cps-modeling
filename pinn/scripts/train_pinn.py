#!/usr/bin/env python3
"""
Training script for PINN.

Usage (from thesis directory):
    python pinn/scripts/train_pinn.py --config configs/pinn_default.yaml
    python pinn/scripts/train_pinn.py --config configs/pinn_default.yaml --checkpoint experiments/pinn_run/best.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from datetime import datetime

# Add project root to path (thesis directory, not pinn)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinn.models import PINN
from pinn.data import create_data_loaders, Normalizer
from pinn.losses import CombinedLoss
from pinn.training import Trainer, CurriculumScheduler, create_curriculum


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str) -> PINN:
    """Create PINN model from configuration."""
    model_config = config.get('model', {})

    model = PINN(
        param_input_dim=model_config.get('param_input_dim', 8),
        param_hidden_dims=model_config.get('param_hidden_dims', [64, 128, 256]),
        param_output_dim=model_config.get('param_output_dim', 128),
        coord_input_dim=model_config.get('coord_input_dim', 2),
        num_fourier_features=model_config.get('num_fourier_features', 128),
        fourier_sigma=model_config.get('fourier_sigma', 4.0),
        coord_hidden_dims=model_config.get('coord_hidden_dims', [128, 256]),
        coord_output_dim=model_config.get('coord_output_dim', 128),
        time_num_frequencies=model_config.get('time_num_frequencies', 8),
        time_hidden_dims=model_config.get('time_hidden_dims', [32]),
        time_output_dim=model_config.get('time_output_dim', 64),
        fusion_hidden_dims=model_config.get('fusion_hidden_dims', [256, 256]),
        fusion_output_dim=model_config.get('fusion_output_dim', 128),
        fusion_use_skip=model_config.get('fusion_use_skip', True),
        field_hidden_dim=model_config.get('field_hidden_dim', 64),
        scalar_hidden_dims=model_config.get('scalar_hidden_dims', [128, 64]),
        num_scalars=model_config.get('num_scalars', 25),
        activation=model_config.get('activation', 'gelu'),
        dropout=model_config.get('dropout', 0.0)
    )

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Train PINN for Cathodic Protection System')
    parser.add_argument('--config', type=str, default='configs/pinn_default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--no-physics', action='store_true',
                        help='Disable physics loss (data-only training)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Override with command line arguments
    if args.device:
        config['device'] = args.device
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.output_dir:
        config['output']['checkpoint_dir'] = args.output_dir
    if args.no_physics:
        config['training']['compute_physics_loss'] = False

    # Device
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"PINN Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # Create output directory
    output_dir = config['output']['checkpoint_dir']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{output_dir}_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config to output directory
    with open(Path(output_dir) / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create data loaders
    print("\nLoading data...")
    data_config = config.get('data', {})
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        h5_path=data_config.get('h5_path', '/home/hhom220/thesis/dataset/cp_dataset_full-10000.h5'),
        batch_size=data_config.get('batch_size', 64),
        train_ratio=data_config.get('train_ratio', 0.8),
        val_ratio=data_config.get('val_ratio', 0.1),
        num_workers=data_config.get('num_workers', 4),
        normalize=data_config.get('normalize', True),
        preload=data_config.get('preload', False),
        seed=data_config.get('seed', 42),
        device=device
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create loss function
    loss_config = config.get('loss', {})
    loss_fn = CombinedLoss(
        lambda_data=loss_config.get('lambda_data', 1.0),
        lambda_pde=loss_config.get('lambda_pde', 0.1),
        lambda_bc=loss_config.get('lambda_bc', 0.5),
        lambda_scalar=loss_config.get('lambda_scalar', 0.1),
        use_relative_data_loss=loss_config.get('use_relative_data_loss', True),
        use_variable_sigma=loss_config.get('use_variable_sigma', True),
        adaptive_weighting=loss_config.get('adaptive_weighting', False)
    )

    # Create curriculum scheduler
    curriculum_config = config.get('curriculum', {})
    curriculum = create_curriculum(curriculum_config, config['training']['max_epochs'])

    # Create optimizer
    training_config = config.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4),
        betas=tuple(training_config.get('betas', [0.9, 0.999]))
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        curriculum=curriculum,
        optimizer=optimizer,
        device=device,
        max_epochs=training_config.get('max_epochs', 150),
        grad_clip=training_config.get('grad_clip', 1.0),
        use_amp=training_config.get('use_amp', True),
        checkpoint_dir=output_dir,
        log_interval=training_config.get('log_interval', 10),
        val_interval=training_config.get('val_interval', 1),
        save_interval=training_config.get('save_interval', 10),
        early_stopping_patience=training_config.get('early_stopping_patience', 20),
        compute_physics_loss=training_config.get('compute_physics_loss', True)
    )

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    print("\nStarting training...")
    history = trainer.train()

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    from pinn.evaluation import compute_metrics

    test_metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        normalizer=normalizer,
        device=device,
        compute_physics=False
    )

    print("\nTest Set Metrics:")
    print(f"  Relative L2 Error: {test_metrics['rel_l2_error']:.4f}")
    print(f"  Max Error: {test_metrics['max_error']:.4f} V")
    print(f"  MAE: {test_metrics['mae']:.4f} V")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Save test metrics
    import json
    with open(Path(output_dir) / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
