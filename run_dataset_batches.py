#!/usr/bin/env python3
"""
Helper script to generate large datasets in batches to avoid OOM.

Due to ~30 MB/case memory leak in underlying libraries, generating
1000 cases requires ~30 GB RAM. This script runs batches of 200 cases
(~6 GB each) and restarts between batches.

Features:
- Automatic resume detection (checks HDF5 file for completed cases)
- Graceful interrupt handling (Ctrl+C saves progress)
- Progress tracking across batches

Usage:
    # Start fresh or auto-resume
    python3 run_dataset_batches.py

    # Manually specify starting case
    python3 run_dataset_batches.py 400

    # Change total cases or batch size
    python3 run_dataset_batches.py --total 5000 --batch 250
"""

import sys
import subprocess
import argparse
import os

def get_completed_count(dataset_file='./dataset/cp_dataset_full.h5'):
    """Check how many cases are already completed."""
    if not os.path.exists(dataset_file):
        return 0

    try:
        import h5py
        with h5py.File(dataset_file, 'r') as f:
            if 'parameters' in f:
                return len(f['parameters'].keys())
    except:
        pass
    return 0

def run_batch(total_cases, batch_size, start_from=None):
    """Run a single batch of dataset generation."""
    print(f"\n{'='*70}")
    if start_from is None:
        print(f"Running batch with auto-resume")
    else:
        print(f"Running batch: cases {start_from} to {start_from + batch_size - 1}")
    print(f"{'='*70}\n")

    # Run parametric_study.py with auto-resume
    if start_from is None:
        code = f"""
from parametric_study import CPS_DatasetGenerator
generator = CPS_DatasetGenerator()
generator.generate_and_save(num_cases={total_cases}, batch_size={batch_size})
"""
    else:
        code = f"""
from parametric_study import CPS_DatasetGenerator
generator = CPS_DatasetGenerator()
generator.generate_and_save(num_cases={total_cases}, batch_size={batch_size}, start_from={start_from})
"""

    # Execute in fresh Python process (clears memory)
    result = subprocess.run([sys.executable, '-c', code], cwd='/home/hhom220/thesis')

    # Exit codes: 0 = success, 1 = Ctrl+C (graceful), 2+ = error
    if result.returncode == 0:
        return True
    elif result.returncode == 1:
        print(f"\n⚠️  Interrupted by user. Progress saved.")
        return True  # Graceful shutdown, can resume
    else:
        print(f"\n❌ Batch failed with exit code {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate dataset in batches')
    parser.add_argument('start_from', nargs='?', type=int, default=None,
                       help='Case index to start from (auto-detect if not specified)')
    parser.add_argument('--total', type=int, default=1000,
                       help='Total number of cases to generate (default: 1000)')
    parser.add_argument('--batch', type=int, default=200,
                       help='Batch size (default: 200)')

    args = parser.parse_args()

    total_cases = args.total
    batch_size = args.batch
    start_from = args.start_from

    # Check current progress
    completed = get_completed_count()
    if completed > 0:
        print(f"📁 Found existing dataset with {completed} completed cases")

    # Auto-detect start if not specified
    if start_from is None and completed > 0:
        start_from = (completed // batch_size) * batch_size
        print(f"   Auto-resuming from case {start_from}")
    elif start_from is None:
        start_from = 0

    # Run batches
    current = start_from
    total_batches = (total_cases + batch_size - 1) // batch_size

    while current < total_cases:
        batch_num = (current // batch_size) + 1

        print(f"\n{'#'*70}")
        print(f"#  BATCH {batch_num}/{total_batches}")
        print(f"#  Progress: {completed}/{total_cases} cases ({100*completed//total_cases}%)")
        print(f"{'#'*70}")

        prev_completed = get_completed_count()

        if not run_batch(total_cases, batch_size, current):
            new_completed = get_completed_count()
            print(f"\n⚠️  Batch {batch_num} interrupted.")
            print(f"    Completed: {new_completed}/{total_cases} cases")
            print(f"    To resume, run:")
            print(f"    python3 run_dataset_batches.py")
            sys.exit(0)  # Graceful exit for resume

        # Update progress
        completed = get_completed_count()
        new_cases = completed - prev_completed
        print(f"\n✅ Batch {batch_num} complete: +{new_cases} cases")

        # Check if we're done
        if completed >= total_cases:
            break

        current += batch_size

    # Final summary
    final_count = get_completed_count()
    print(f"\n{'='*70}")
    print("✅ DATASET GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total cases: {final_count}/{total_cases}")
    print(f"Dataset saved to: ./dataset/cp_dataset_full.h5")

if __name__ == "__main__":
    main()
