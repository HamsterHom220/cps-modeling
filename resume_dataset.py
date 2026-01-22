#!/usr/bin/env python3
"""
Quick script to resume dataset generation after interruption.

This script automatically:
- Detects where to resume from
- Runs multiple batches with memory clearing between them
- Continues until all cases are generated

Usage:
    python3 resume_dataset.py

Or specify custom parameters:
    python3 resume_dataset.py --total 5000 --batch 250
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

def run_single_batch(total_cases, batch_size):
    """Run a single batch in a fresh Python process."""
    code = f"""
from parametric_study import CPS_DatasetGenerator
generator = CPS_DatasetGenerator()
generator.generate_and_save(
    num_cases={total_cases},
    batch_size={batch_size},
    start_from=None,      # Auto-detect
    auto_resume=True       # Skip completed cases
)
"""
    # Execute in fresh Python process (clears memory)
    result = subprocess.run([sys.executable, '-c', code], cwd='/home/hhom220/thesis')

    # Exit codes: 0 = success, 1 = graceful interrupt
    return result.returncode in [0, 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume dataset generation')
    parser.add_argument('--total', type=int, default=1000,
                       help='Total number of cases to generate (default: 1000)')
    parser.add_argument('--batch', type=int, default=200,
                       help='Batch size (default: 200)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("🔄 AUTO-RESUME DATASET GENERATION")
    print(f"{'='*70}")
    print(f"Target: {args.total} total cases")
    print(f"Batch size: {args.batch} cases")
    print(f"Strategy: Run multiple batches with memory clearing between them")
    print(f"{'='*70}\n")

    batch_num = 1
    total_batches = (args.total + args.batch - 1) // args.batch

    while True:
        # Check current progress
        completed = get_completed_count()

        if completed >= args.total:
            print(f"\n{'='*70}")
            print("✅ DATASET GENERATION COMPLETE!")
            print(f"{'='*70}")
            print(f"Total cases: {completed}/{args.total}")
            print(f"Dataset saved to: ./dataset/cp_dataset_full.h5")
            break

        # Calculate batch number
        current_batch = (completed // args.batch) + 1
        remaining = args.total - completed

        print(f"\n{'#'*70}")
        print(f"#  BATCH {current_batch}/{total_batches}")
        print(f"#  Progress: {completed}/{args.total} cases ({100*completed//args.total}%)")
        print(f"#  Remaining: {remaining} cases")
        print(f"{'#'*70}\n")

        # Run one batch
        prev_completed = completed

        if not run_single_batch(args.total, args.batch):
            # Batch failed or was interrupted
            new_completed = get_completed_count()
            new_cases = new_completed - prev_completed

            print(f"\n⚠️  Batch {current_batch} interrupted.")
            print(f"    Generated {new_cases} cases in this batch")
            print(f"    Total: {new_completed}/{args.total} cases")
            print(f"\n   To resume, run:")
            print(f"   python3 resume_dataset.py")
            sys.exit(0)

        # Check progress after batch
        new_completed = get_completed_count()
        new_cases = new_completed - prev_completed

        print(f"\n✅ Batch {current_batch} complete: +{new_cases} cases")
        print(f"   Total progress: {new_completed}/{args.total} ({100*new_completed//args.total}%)")

        batch_num += 1

    print(f"\n🎉 All {args.total} cases generated successfully!")
