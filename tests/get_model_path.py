#!/usr/bin/env python3
"""Helper script to get model path for smoke test."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Dietnet.model_manager import get_model_path

if __name__ == '__main__':
    preset = sys.argv[1] if len(sys.argv) > 1 else '1kgp_default'
    try:
        model_path = get_model_path(preset)
        print(model_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
