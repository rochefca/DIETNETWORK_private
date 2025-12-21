#!/bin/bash
# Package 1KGP test data for distribution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source data paths
PLINK_PREFIX="/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
LABELS="/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/labels_pop_subsampleV1.tsv"

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
OUTPUT="dietnet_1kgp_test_data_v1.tar.gz"
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
