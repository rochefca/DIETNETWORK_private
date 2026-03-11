#!/bin/bash
# Download the 1KGP default pretrained model into the local DietNetwork cache.
# Useful when compute nodes lack internet access; run this on a login node first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
# Allow empty PYTHONPATH when running with 'set -u'
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Default cache location used by DietNetwork (~/.cache/dietnet)
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/dietnet"
PRESET="1kgp_default"

echo "=========================================="
echo "Downloading DietNetwork pretrained model ($PRESET)"
echo "Cache dir: $CACHE_ROOT"
echo "=========================================="

python3 - "$CACHE_ROOT" "$PRESET" << 'PYTHON_EOF'
import hashlib
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path

cache_root = Path(sys.argv[1])
preset = sys.argv[2]

# Lazy import from project root (already on PYTHONPATH)
from Dietnet.pretrained_models import PRETRAINED_MODELS

if preset not in PRETRAINED_MODELS:
    print(f"ERROR: Unknown preset '{preset}'.", file=sys.stderr)
    sys.exit(1)

config = PRETRAINED_MODELS[preset]
dest = cache_root / config["cache_dir"]

if dest.exists():
    print(f"✓ Model already cached at {dest}")
    sys.exit(0)

model_url = config.get("model_url")
if model_url in (None, "PLACEHOLDER_MODEL_URL"):
    print("ERROR: Model URL not configured in Dietnet/pretrained_models.py", file=sys.stderr)
    sys.exit(1)

cache_root.mkdir(parents=True, exist_ok=True)
temp_file = cache_root / f"{preset}_temp.tar.gz"

print(f"Downloading from {model_url}...")
try:
    urllib.request.urlretrieve(model_url, temp_file)
except Exception as e:
    if temp_file.exists():
        temp_file.unlink()
    print(f"ERROR: Failed to download model: {e}", file=sys.stderr)
    sys.exit(1)

expected_hash = config.get("model_sha256")
if expected_hash and expected_hash != "PLACEHOLDER_SHA256":
    print("Validating checksum...")
    h = hashlib.sha256()
    with open(temp_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash != expected_hash:
        temp_file.unlink()
        print("ERROR: Checksum mismatch!", file=sys.stderr)
        print(f"Expected: {expected_hash}", file=sys.stderr)
        print(f"Got:      {actual_hash}", file=sys.stderr)
        sys.exit(1)
    print("✓ Checksum valid")

print(f"Extracting to {dest}...")
dest.mkdir(parents=True, exist_ok=True)
try:
    with tarfile.open(temp_file, "r:gz") as tar:
        tar.extractall(dest)
except Exception as e:
    shutil.rmtree(dest, ignore_errors=True)
    temp_file.unlink()
    print(f"ERROR: Failed to extract model: {e}", file=sys.stderr)
    sys.exit(1)

temp_file.unlink()
print(f"✓ Model cached at {dest}")
PYTHON_EOF

echo ""
echo "Done. You can now run smoke tests on nodes without internet access."
