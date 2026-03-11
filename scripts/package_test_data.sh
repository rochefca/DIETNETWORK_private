#!/bin/bash
# Package 1KGP test data for distribution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# User configuration (edit these or pass as environment variables)
PLINK_PREFIX="${PLINK_PREFIX:-/path/to/plink_prefix}"
LABELS="${LABELS:-/path/to/labels.tsv}"
OUTPUT="${OUTPUT:-dietnet_1kgp_test_data_v1.tar.gz}"

echo "=========================================="
echo "Packaging DietNetwork 1KGP Test Data"
echo "=========================================="

# Check source files exist
if [ ! -f "${PLINK_PREFIX}.bed" ]; then
    echo "ERROR: PLINK files not found at ${PLINK_PREFIX}.*"
    exit 1
fi

if [ ! -f "$LABELS" ]; then
    echo "ERROR: Labels file not found at $LABELS"
    exit 1
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/1kgp_test_data"

echo ""
echo "Copying files to temporary directory..."

# Copy PLINK files
echo "  - PLINK .bed file..."
cp "${PLINK_PREFIX}.bed" "$TEMP_DIR/1kgp_test_data/"
echo "  - PLINK .bim file..."
cp "${PLINK_PREFIX}.bim" "$TEMP_DIR/1kgp_test_data/"
echo "  - PLINK .fam file..."
cp "${PLINK_PREFIX}.fam" "$TEMP_DIR/1kgp_test_data/"

# Copy labels
echo "  - Labels file..."
cp "$LABELS" "$TEMP_DIR/1kgp_test_data/"

# Create tarball
echo ""
echo "Creating tarball..."
cd "$TEMP_DIR"
tar -czf "$OUTPUT" 1kgp_test_data/

# Move to project root
mv "$OUTPUT" "$PROJECT_ROOT/"

# Compute SHA256
cd "$PROJECT_ROOT"
SHA256=$(sha256sum "$OUTPUT" | awk '{print $1}')

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "✓ Package created: $OUTPUT"
echo "  Size: $(du -h "$OUTPUT" | awk '{print $1}')"
echo "  SHA256: $SHA256"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT to Zenodo/Figshare/institutional server"
echo "2. Update Dietnet/pretrained_models.py TEST_DATA with:"
echo "   - plink_url: '<your_download_url>'"
echo "   - plink_sha256: '$SHA256'"
echo ""
echo "Example:"
echo "  \"plink_url\": \"https://zenodo.org/record/XXXXX/files/$OUTPUT\","
echo "  \"plink_sha256\": \"$SHA256\","
