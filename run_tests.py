#!/usr/bin/env python
"""Run tests without needing to set PYTHONPATH"""
import sys
import os
from pathlib import Path

# Force UTF-8 stdout so unicode symbols (✓ ✗ ✅ ❌ ⚠) used by tests render on
# Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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
        "tests/test_maps.py",
        "tests/test_action_decoder_parity.py",
    ]

    print("Running all tests...\n")

    # Propagate UTF-8 stdio to subprocesses so their prints don't hit cp1252.
    env = {
        'PYTHONPATH': str(project_root),
        'PYTHONIOENCODING': 'utf-8',
    }
    for test in tests:
        print(f"{'='*60}")
        print(f"Running {test}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, test], cwd=project_root, env={**os.environ, **env})
        if result.returncode != 0:
            print(f"\n❌ {test} FAILED")
            sys.exit(1)
        print()

    print(f"{'='*60}")
    print("✅ ALL TESTS PASSED")
    print(f"{'='*60}")
