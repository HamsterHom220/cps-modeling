#!/usr/bin/env python3
"""
Quick script to resume dataset generation after interruption.

This is a convenience wrapper that automatically detects where to resume from
and continues generation.

Usage:
    python3 resume_dataset.py

Or specify custom parameters:
    python3 resume_dataset.py --total 5000 --batch 250
"""

from parametric_study import CPS_DatasetGenerator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume dataset generation')
    parser.add_argument('--total', type=int, default=1000,
                       help='Total number of cases to generate (default: 1000)')
    parser.add_argument('--batch', type=int, default=200,
                       help='Batch size (default: 200)')

    args = parser.parse_args()

    print("🔄 Resuming dataset generation with auto-detection...")
    print(f"   Target: {args.total} total cases")
    print(f"   Batch size: {args.batch} cases\n")

    generator = CPS_DatasetGenerator()

    # Auto-detect resume point and skip completed cases
    generator.generate_and_save(
        num_cases=args.total,
        batch_size=args.batch,
        start_from=None,      # Auto-detect
        auto_resume=True       # Skip completed cases
    )

    print("\n✅ Session complete!")
    completed = len(generator.get_completed_cases())
    print(f"   Total cases in dataset: {completed}/{args.total}")

    if completed < args.total:
        print(f"\n   To continue, restart this script:")
        print(f"   python3 resume_dataset.py")
