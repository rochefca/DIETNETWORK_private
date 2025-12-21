#!/bin/bash
# Download 1KGP test data for smoke testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/smoke_test_config.sh"

mkdir -p "$TEST_DATA_DIR"

echo "=========================================="
echo "Downloading DietNetwork Test Data"
echo "=========================================="

# Import test data config from Python and download
python3 - "$TEST_DATA_DIR" << 'PYTHON_EOF'
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Dietnet.pretrained_models import TEST_DATA
import urllib.request
import tarfile
import hashlib

config = TEST_DATA['1kgp_default']
test_data_dir = Path(sys.argv[1])

# Check if already exists
plink_base = config['plink_basename']
if (test_data_dir / f"{plink_base}.bed").exists():
    print("✓ Test data already downloaded")
    sys.exit(0)

# Check if URL is configured
if config['plink_url'] == 'PLACEHOLDER_PLINK_URL':
    # URL not configured - try to use local files from shared storage
    print("⚠ Test data URL not configured yet, checking for local files...")
    local_path = Path("/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1")
    local_plink = local_path / plink_base
    local_labels = local_path / "labels_pop_subsampleV1.tsv"

    if local_plink.with_suffix('.bed').exists():
        print(f"✓ Using local test data from {local_path}")
        # Create symlinks to local files
        import os
        test_data_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['.bed', '.bim', '.fam']:
            src = local_plink.with_suffix(ext)
            dst = test_data_dir / f"{plink_base}{ext}"
            if not dst.exists():
                os.symlink(src, dst)
        # Symlink labels
        dst_labels = test_data_dir / "labels_pop_subsampleV1.tsv"
        if not dst_labels.exists():
            os.symlink(local_labels, dst_labels)
        print("✓ Test data linked successfully")
        sys.exit(0)
    else:
        print("ERROR: Test data URL not configured and local files not found!")
        print(f"Looked in: {local_path}")
        print("Please update Dietnet/pretrained_models.py with the actual download URL.")
        sys.exit(1)

# Download
print(f"Downloading from {config['plink_url']}...")
temp_file = test_data_dir.parent / "test_data_temp.tar.gz"

try:
    urllib.request.urlretrieve(config['plink_url'], temp_file)
except Exception as e:
    if temp_file.exists():
        temp_file.unlink()
    print(f"ERROR: Failed to download test data: {e}")
    sys.exit(1)

# Validate checksum
if config.get('plink_sha256') and config['plink_sha256'] != 'PLACEHOLDER_SHA256':
    print("Validating checksum...")
    with open(temp_file, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != config['plink_sha256']:
        temp_file.unlink()
        print(f"ERROR: Checksum mismatch!")
        print(f"Expected: {config['plink_sha256']}")
        print(f"Got:      {actual_hash}")
        sys.exit(1)
    print("✓ Checksum valid")

# Extract
print("Extracting...")
try:
    with tarfile.open(temp_file, 'r:gz') as tar:
        tar.extractall(test_data_dir.parent)
except Exception as e:
    temp_file.unlink()
    print(f"ERROR: Failed to extract test data: {e}")
    sys.exit(1)

temp_file.unlink()
print("✓ Test data download complete")
PYTHON_EOF

echo ""
echo "✓ Test data ready at $TEST_DATA_DIR"
