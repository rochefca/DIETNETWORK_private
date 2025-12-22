#!/bin/bash
# Configuration for smoke tests

# Model cache location
export MODEL_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/dietnet"

# Test data location (relative to tests/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_DATA_DIR="$SCRIPT_DIR/data/1kgp_test_data"

# Expected accuracy range for 1KGP smoke test
export EXPECTED_ACCURACY_MIN=85
export EXPECTED_ACCURACY_MAX=100
