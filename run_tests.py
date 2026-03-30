#!/usr/bin/env python
"""Run tests without needing to set PYTHONPATH"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import subprocess

    tests = [
        "tests/test_mechanics.py",
        "tests/test_combat.py",
        "tests/test_regions.py",
        "tests/test_run.py",
    ]

    print("Running all tests...\n")

    for test in tests:
        print(f"{'='*60}")
        print(f"Running {test}")
        print(f"{'='*60}")
        # Set PYTHONPATH so subprocess can find parallel_risk module
        env = {'PYTHONPATH': str(project_root)}
        result = subprocess.run([sys.executable, test], cwd=project_root, env={**os.environ, **env})
        if result.returncode != 0:
            print(f"\n❌ {test} FAILED")
            sys.exit(1)
        print()

    print(f"{'='*60}")
    print("✅ ALL TESTS PASSED")
    print(f"{'='*60}")
