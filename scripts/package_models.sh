#!/bin/bash
# Package pretrained_1000g for distribution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# User configuration
SOURCE_DIR="${1:-pretrained_1000g_models}"   # pass as arg or set here
OUTPUT="${2:-dietnet_1kgp_default_v1.tar.gz}"

echo "=========================================="
echo "Packaging DietNetwork 1KGP Model"
echo "=========================================="

cd "$PROJECT_ROOT"

# Check source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: $SOURCE_DIR/ not found"
    echo "Expected location: $PROJECT_ROOT/$SOURCE_DIR"
    echo "This should contain all 15 models (3 seeds × 5 folds)"
    exit 1
fi

# Count models
N_MODELS=$(find "$SOURCE_DIR" -name "metadata.json" | wc -l)
echo "Found $N_MODELS models in $SOURCE_DIR/"

if [ "$N_MODELS" -ne 15 ]; then
    echo "WARNING: Expected 15 models (3 seeds × 5 folds), found $N_MODELS"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Create tarball
echo "Creating $OUTPUT..."

# Create tarball from inside the directory so it extracts without the parent folder
cd "$SOURCE_DIR"
tar -czf "../$OUTPUT" seed_*
cd ..

# Compute SHA256
SHA256=$(sha256sum "$OUTPUT" | awk '{print $1}')

echo ""
echo "✓ Package created: $OUTPUT"
echo "  Size: $(du -h "$OUTPUT" | awk '{print $1}')"
echo "  SHA256: $SHA256"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT to Zenodo/Figshare/institutional server"
echo "2. Update Dietnet/pretrained_models.py with:"
echo "   - model_url: '<your_download_url>'"
echo "   - model_sha256: '$SHA256'"
echo ""
echo "Example:"
echo "  \"model_url\": \"https://zenodo.org/record/XXXXX/files/$OUTPUT\","
echo "  \"model_sha256\": \"$SHA256\","
