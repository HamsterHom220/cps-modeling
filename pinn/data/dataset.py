"""
HDF5 Dataset loader for PINN training.

Loads the CP dataset with lazy loading for memory efficiency.
Supports splitting by case (not snapshot) for proper train/val/test splits.
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .normalization import Normalizer


class CPDataset(Dataset):
    """
    PyTorch Dataset for Cathodic Protection System data.

    Supports:
    - Lazy loading from HDF5 (memory efficient)
    - Case-based indexing (each case has 7 time points)
    - Snapshot-based indexing (70,000 total samples)
    - On-the-fly normalization
    - Coordinate grid generation
    """

    TIME_POINTS = [0, 5, 10, 15, 20, 25, 30]  # years
    GRID_SHAPE = (20, 40)  # (ny, nx)

    # Key scalar metrics to include
    SCALAR_METRICS = [
        'coverage', 'avg_potential', 'min_potential', 'max_potential',
        'current', 'current_density', 'corrosion_rate',
        'anode_resistance', 'pipe_resistance', 'coating_resistance',
        'voltage_drop', 'polarization_loss', 'soil_drop',
        'protection_current', 'protection_density',
        'anodic_current', 'cathodic_current',
        'total_anode_current', 'avg_anode_potential',
        'flux_anode', 'flux_pipe', 'flux_balance',
        'newton_iterations', 'newton_converged', 'residual_norm'
    ]

    def __init__(
        self,
        h5_path: str,
        case_indices: Optional[List[int]] = None,
        normalizer: Optional[Normalizer] = None,
        normalize: bool = True,
        mode: str = 'snapshot',
        include_sigma: bool = True,
        include_scalars: bool = True,
        preload: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize dataset.

        Args:
            h5_path: Path to HDF5 file
            case_indices: List of case indices to use (for train/val/test split)
            normalizer: Normalizer instance (created if None)
            normalize: Whether to normalize data
            mode: 'snapshot' for individual samples, 'case' for full time series
            include_sigma: Whether to include conductivity field
            include_scalars: Whether to include scalar metrics
            preload: Whether to load all data into memory
            device: Device for tensors
        """
        self.h5_path = Path(h5_path)
        self.normalize = normalize
        self.mode = mode
        self.include_sigma = include_sigma
        self.include_scalars = include_scalars
        self.preload = preload
        self.device = device

        # Open file to get metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.total_cases = len(f['parameters'])
            # Get grid shape from first case
            first_case = list(f['fields'].keys())[0]
            first_time = list(f['fields'][first_case].keys())[0]
            self.grid_shape = f['fields'][first_case][first_time]['phi'].shape

        # Case indices to use
        if case_indices is None:
            self.case_indices = list(range(self.total_cases))
        else:
            self.case_indices = case_indices

        self.n_cases = len(self.case_indices)
        self.n_time_points = len(self.TIME_POINTS)
        self.n_snapshots = self.n_cases * self.n_time_points

        # Normalizer
        self.normalizer = normalizer if normalizer is not None else Normalizer(device=device)

        # Generate coordinate grid
        self._init_coordinate_grid()

        # Preload data if requested
        self._preloaded_data = None
        if preload:
            self._preload_all_data()

    def _init_coordinate_grid(self):
        """Initialize normalized coordinate grid."""
        ny, nx = self.grid_shape

        # Physical coordinates (domain: 20m x 8m)
        x = np.linspace(0, 20, nx)
        y = np.linspace(0, 8, ny)
        xx, yy = np.meshgrid(x, y)

        # Store as tensors
        self.x_grid = torch.tensor(xx, dtype=torch.float32)
        self.y_grid = torch.tensor(yy, dtype=torch.float32)

        # Normalized coordinates
        x_norm, y_norm = self.normalizer.normalize_coords(self.x_grid, self.y_grid)
        self.x_grid_norm = x_norm
        self.y_grid_norm = y_norm

        # Flattened coordinates for point-wise queries
        self.coords_flat = torch.stack([
            self.x_grid.flatten(),
            self.y_grid.flatten()
        ], dim=-1)  # (800, 2)

        self.coords_flat_norm = torch.stack([
            self.x_grid_norm.flatten(),
            self.y_grid_norm.flatten()
        ], dim=-1)  # (800, 2)

    def _preload_all_data(self):
        """Load all data into memory for faster access."""
        print(f"Preloading {self.n_snapshots} snapshots into memory...")
        self._preloaded_data = {
            'params': [],
            'phi': [],
            'sigma': [] if self.include_sigma else None,
            'scalars': [] if self.include_scalars else None,
            'time': [],
            'case_idx': [],
        }

        with h5py.File(self.h5_path, 'r') as f:
            for case_idx in self.case_indices:
                case_key = f'case_{case_idx:04d}'
                params = f['parameters'][case_key][:]

                for t_idx, t in enumerate(self.TIME_POINTS):
                    time_key = f't_{t_idx:03d}'

                    # Load fields
                    phi = f['fields'][case_key][time_key]['phi'][:]
                    self._preloaded_data['phi'].append(phi)

                    if self.include_sigma:
                        sigma = f['fields'][case_key][time_key]['sigma'][:]
                        self._preloaded_data['sigma'].append(sigma)

                    # Load scalars
                    if self.include_scalars:
                        result_attrs = dict(f['results'][case_key][time_key].attrs)
                        scalars = {k: result_attrs.get(k, 0.0) for k in self.SCALAR_METRICS}
                        self._preloaded_data['scalars'].append(scalars)

                    self._preloaded_data['params'].append(params)
                    self._preloaded_data['time'].append(t)
                    self._preloaded_data['case_idx'].append(case_idx)

        # Convert to arrays/tensors
        self._preloaded_data['params'] = np.array(self._preloaded_data['params'])
        self._preloaded_data['phi'] = np.array(self._preloaded_data['phi'])
        if self.include_sigma:
            self._preloaded_data['sigma'] = np.array(self._preloaded_data['sigma'])
        self._preloaded_data['time'] = np.array(self._preloaded_data['time'])
        self._preloaded_data['case_idx'] = np.array(self._preloaded_data['case_idx'])

        print(f"Preloaded data shapes: params={self._preloaded_data['params'].shape}, "
              f"phi={self._preloaded_data['phi'].shape}")

    def _get_case_key(self, case_idx: int) -> str:
        """Get HDF5 key for case index."""
        return f'case_{case_idx:04d}'

    def _get_time_key(self, t_idx: int) -> str:
        """Get HDF5 key for time index."""
        return f't_{t_idx:03d}'

    def _snapshot_to_case_time(self, snapshot_idx: int) -> Tuple[int, int]:
        """Convert snapshot index to (case_index, time_index)."""
        case_local_idx = snapshot_idx // self.n_time_points
        t_idx = snapshot_idx % self.n_time_points
        case_idx = self.case_indices[case_local_idx]
        return case_idx, t_idx

    def __len__(self) -> int:
        if self.mode == 'snapshot':
            return self.n_snapshots
        else:  # case mode
            return self.n_cases

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'snapshot':
            return self._get_snapshot(idx)
        else:
            return self._get_case(idx)

    def _get_snapshot(self, snapshot_idx: int) -> Dict[str, torch.Tensor]:
        """Get single snapshot by index."""
        if self._preloaded_data is not None:
            return self._get_snapshot_preloaded(snapshot_idx)

        case_idx, t_idx = self._snapshot_to_case_time(snapshot_idx)
        case_key = self._get_case_key(case_idx)
        time_key = self._get_time_key(t_idx)
        t = self.TIME_POINTS[t_idx]

        with h5py.File(self.h5_path, 'r') as f:
            # Load parameters
            params = torch.tensor(f['parameters'][case_key][:], dtype=torch.float32)

            # Load potential field
            phi = torch.tensor(f['fields'][case_key][time_key]['phi'][:], dtype=torch.float32)

            # Load conductivity if needed
            sigma = None
            if self.include_sigma:
                sigma = torch.tensor(f['fields'][case_key][time_key]['sigma'][:], dtype=torch.float32)

            # Load scalars if needed
            scalars = None
            if self.include_scalars:
                result_attrs = dict(f['results'][case_key][time_key].attrs)
                scalars = {k: torch.tensor(result_attrs.get(k, 0.0), dtype=torch.float32)
                           for k in self.SCALAR_METRICS if k in result_attrs}

        # Build output dict
        return self._build_output_dict(params, phi, sigma, scalars, t, case_idx)

    def _get_snapshot_preloaded(self, snapshot_idx: int) -> Dict[str, torch.Tensor]:
        """Get snapshot from preloaded data."""
        params = torch.tensor(self._preloaded_data['params'][snapshot_idx], dtype=torch.float32)
        phi = torch.tensor(self._preloaded_data['phi'][snapshot_idx], dtype=torch.float32)

        sigma = None
        if self.include_sigma:
            sigma = torch.tensor(self._preloaded_data['sigma'][snapshot_idx], dtype=torch.float32)

        scalars = None
        if self.include_scalars:
            scalars = {k: torch.tensor(v, dtype=torch.float32)
                       for k, v in self._preloaded_data['scalars'][snapshot_idx].items()}

        t = self._preloaded_data['time'][snapshot_idx]
        case_idx = self._preloaded_data['case_idx'][snapshot_idx]

        return self._build_output_dict(params, phi, sigma, scalars, t, case_idx)

    def _get_case(self, case_local_idx: int) -> Dict[str, torch.Tensor]:
        """Get full time series for a case."""
        case_idx = self.case_indices[case_local_idx]
        case_key = self._get_case_key(case_idx)

        with h5py.File(self.h5_path, 'r') as f:
            # Load parameters (same for all time points)
            params = torch.tensor(f['parameters'][case_key][:], dtype=torch.float32)

            # Load fields for all time points
            phi_list = []
            sigma_list = []
            scalars_list = []

            for t_idx in range(self.n_time_points):
                time_key = self._get_time_key(t_idx)

                phi_list.append(f['fields'][case_key][time_key]['phi'][:])

                if self.include_sigma:
                    sigma_list.append(f['fields'][case_key][time_key]['sigma'][:])

                if self.include_scalars:
                    result_attrs = dict(f['results'][case_key][time_key].attrs)
                    scalars_list.append({k: result_attrs.get(k, 0.0)
                                         for k in self.SCALAR_METRICS if k in result_attrs})

        # Stack time dimension
        phi = torch.tensor(np.stack(phi_list), dtype=torch.float32)  # (7, 20, 40)
        sigma = torch.tensor(np.stack(sigma_list), dtype=torch.float32) if self.include_sigma else None
        times = torch.tensor(self.TIME_POINTS, dtype=torch.float32)

        # Stack scalars
        scalars = None
        if self.include_scalars:
            scalars = {}
            for k in scalars_list[0].keys():
                scalars[k] = torch.tensor([s[k] for s in scalars_list], dtype=torch.float32)

        return self._build_case_output_dict(params, phi, sigma, scalars, times, case_idx)

    def _build_output_dict(
        self,
        params: torch.Tensor,
        phi: torch.Tensor,
        sigma: Optional[torch.Tensor],
        scalars: Optional[Dict[str, torch.Tensor]],
        t: float,
        case_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Build output dictionary for single snapshot."""
        t_tensor = torch.tensor(t, dtype=torch.float32)

        # Normalize if requested
        if self.normalize:
            params = self.normalizer.normalize_params(params)
            phi = self.normalizer.normalize_phi(phi)
            t_tensor = self.normalizer.normalize_time(t_tensor)
            if sigma is not None:
                sigma = self.normalizer.normalize_sigma(sigma)

        output = {
            'params': params,                    # (8,)
            'phi': phi,                          # (20, 40)
            'time': t_tensor,                    # scalar
            'coords': self.coords_flat_norm if self.normalize else self.coords_flat,  # (800, 2)
            'x_grid': self.x_grid_norm if self.normalize else self.x_grid,  # (20, 40)
            'y_grid': self.y_grid_norm if self.normalize else self.y_grid,  # (20, 40)
            'case_idx': torch.tensor(case_idx, dtype=torch.long),
        }

        if sigma is not None:
            output['sigma'] = sigma  # (20, 40)

        if scalars is not None:
            # Stack scalars into tensor (for loss computation)
            scalar_keys = sorted(scalars.keys())
            output['scalars'] = torch.stack([scalars[k] for k in scalar_keys])
            output['scalar_names'] = scalar_keys

        return output

    def _build_case_output_dict(
        self,
        params: torch.Tensor,
        phi: torch.Tensor,
        sigma: Optional[torch.Tensor],
        scalars: Optional[Dict[str, torch.Tensor]],
        times: torch.Tensor,
        case_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Build output dictionary for full case (time series)."""
        # Normalize if requested
        if self.normalize:
            params = self.normalizer.normalize_params(params)
            phi = self.normalizer.normalize_phi(phi)
            times = self.normalizer.normalize_time(times)
            if sigma is not None:
                sigma = self.normalizer.normalize_sigma(sigma)

        output = {
            'params': params,                    # (8,)
            'phi': phi,                          # (7, 20, 40)
            'time': times,                       # (7,)
            'coords': self.coords_flat_norm if self.normalize else self.coords_flat,
            'x_grid': self.x_grid_norm if self.normalize else self.x_grid,
            'y_grid': self.y_grid_norm if self.normalize else self.y_grid,
            'case_idx': torch.tensor(case_idx, dtype=torch.long),
        }

        if sigma is not None:
            output['sigma'] = sigma  # (7, 20, 40)

        if scalars is not None:
            scalar_keys = sorted(scalars.keys())
            output['scalars'] = torch.stack([scalars[k] for k in scalar_keys], dim=-1)  # (7, n_scalars)
            output['scalar_names'] = scalar_keys

        return output


def create_data_loaders(
    h5_path: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    normalizer: Optional[Normalizer] = None,
    normalize: bool = True,
    mode: str = 'snapshot',
    preload: bool = False,
    seed: int = 42,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader, DataLoader, Normalizer]:
    """
    Create train/val/test data loaders with case-based splits.

    Args:
        h5_path: Path to HDF5 file
        batch_size: Batch size for training
        train_ratio: Fraction of cases for training
        val_ratio: Fraction of cases for validation
        num_workers: Number of data loading workers
        normalizer: Pre-computed normalizer (computed if None)
        normalize: Whether to normalize data
        mode: 'snapshot' or 'case'
        preload: Whether to preload data into memory
        seed: Random seed for reproducibility
        device: Device for normalizer

    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalizer)
    """
    # Get total number of cases
    with h5py.File(h5_path, 'r') as f:
        total_cases = len(f['parameters'])

    # Create case indices
    np.random.seed(seed)
    case_indices = np.random.permutation(total_cases)

    # Split
    n_train = int(total_cases * train_ratio)
    n_val = int(total_cases * val_ratio)

    train_indices = case_indices[:n_train].tolist()
    val_indices = case_indices[n_train:n_train + n_val].tolist()
    test_indices = case_indices[n_train + n_val:].tolist()

    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test cases")

    # Create normalizer if not provided
    if normalizer is None:
        normalizer = Normalizer(device=device)

    # Create datasets
    train_dataset = CPDataset(
        h5_path, case_indices=train_indices, normalizer=normalizer,
        normalize=normalize, mode=mode, preload=preload, device=device
    )
    val_dataset = CPDataset(
        h5_path, case_indices=val_indices, normalizer=normalizer,
        normalize=normalize, mode=mode, preload=preload, device=device
    )
    test_dataset = CPDataset(
        h5_path, case_indices=test_indices, normalizer=normalizer,
        normalize=normalize, mode=mode, preload=preload, device=device
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, normalizer
