#!/usr/bin/env python3
"""
Helper script to generate large datasets in batches to avoid OOM.

Due to ~30 MB/case memory leak in underlying libraries, generating
1000 cases requires ~30 GB RAM. This script runs batches of 200 cases
(~6 GB each) and restarts between batches.

Usage:
    python3 run_dataset_batches.py

Or manually resume from a specific batch:
    python3 run_dataset_batches.py 400  # Start from case 400
"""

import sys
import subprocess

# Configuration
TOTAL_CASES = 1000
BATCH_SIZE = 200  # Safe for ~8 GB RAM systems

def run_batch(start_from=0):
    """Run a single batch of dataset generation."""
    print(f"\n{'='*70}")
    print(f"Running batch: cases {start_from} to {start_from + BATCH_SIZE - 1}")
    print(f"{'='*70}\n")

    # Run parametric_study.py with specific start point
    code = f"""
from parametric_study import CPS_DatasetGenerator
generator = CPS_DatasetGenerator()
generator.generate_and_save(num_cases={TOTAL_CASES}, batch_size={BATCH_SIZE}, start_from={start_from})
"""

    # Execute in fresh Python process (clears memory)
    result = subprocess.run([sys.executable, '-c', code], cwd='/home/hhom220/thesis')

    if result.returncode != 0:
        print(f"\n❌ Batch failed with exit code {result.returncode}")
        return False

    return True

def main():
    # Check if user specified starting point
    if len(sys.argv) > 1:
        start_from = int(sys.argv[1])
    else:
        start_from = 0

    # Run batches
    current = start_from
    batch_num = (start_from // BATCH_SIZE) + 1
    total_batches = (TOTAL_CASES + BATCH_SIZE - 1) // BATCH_SIZE

    while current < TOTAL_CASES:
        print(f"\n{'#'*70}")
        print(f"#  BATCH {batch_num}/{total_batches}")
        print(f"{'#'*70}")

        if not run_batch(current):
            print(f"\n⚠️  Batch {batch_num} failed. Resume with:")
            print(f"    python3 run_dataset_batches.py {current}")
            sys.exit(1)

        current += BATCH_SIZE
        batch_num += 1

    print(f"\n{'='*70}")
    print("✅ ALL BATCHES COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated {TOTAL_CASES} cases in {total_batches} batches")
    print(f"Dataset saved to: ./dataset/cp_dataset_full.h5")

if __name__ == "__main__":
    main()
