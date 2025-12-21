#!/bin/bash
#
# Setup script for DietNetwork
#
# Usage:
#   bash setup.sh
#
# IMPORTANT: Run this on LOGIN NODE (has internet access)
#

set -e  # Exit on error

echo "=========================================="
echo "DietNetwork: Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Must run from DIETNETWORK directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Select Python interpreter
PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        if [ -n "$PYTHON_BIN" ]; then
            uv venv --python "$PYTHON_BIN"
        else
            echo "WARNING: python not found; using default for venv."
            uv venv
        fi
    else
        if [ -n "$PYTHON_BIN" ]; then
            "$PYTHON_BIN" -m venv .venv
        else
            echo "ERROR: python not found. Please load a module or install one, then re-run setup."
            exit 1
        fi
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install package
echo "Installing DietNetwork..."
if command -v uv &> /dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

# Download external tools (CRITICAL: compute nodes have no internet!)
echo ""
echo "=========================================="
echo "Downloading External Tools"
echo "=========================================="
echo ""
echo "IMPORTANT: Compute nodes have no internet access."
echo "Downloading PLINK now..."
echo ""

mkdir -p bin

# Download PLINK2
if [ ! -f "bin/plink2" ]; then
    echo "Downloading PLINK 2.0..."
    PLINK2_URL="https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20241124.zip"

    # Try wget first (better for binary downloads), then curl
    if command -v wget &> /dev/null; then
        wget -O bin/plink2.zip "$PLINK2_URL"
    else
        curl -L -o bin/plink2.zip "$PLINK2_URL"
    fi

    # Extract
    unzip -o bin/plink2.zip -d bin/
    chmod +x bin/plink2
    rm bin/plink2.zip
    # Remove extra files (if any)
    rm -f bin/LICENSE bin/prettify bin/toy.*
    echo "✓ PLINK2 downloaded: $(ls -lh bin/plink2 | awk '{print $5}')"
else
    echo "✓ PLINK2 already exists"
fi

# Set environment variable for PLINK2
export PLINK_PATH="$(pwd)/bin/plink2"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Downloaded tools to bin/:"
ls -lh bin/ 2>/dev/null || echo "  (no files yet)"
echo ""
echo "Total size: $(du -sh bin/ 2>/dev/null | awk '{print $1}' || echo '0')"
echo ""
echo "Next steps:"
echo ""
echo "1. Convert pre-trained models (if needed):"
echo "   python Dietnet/convert_legacy_models.py --help"
echo ""
echo "2. Run inference on test dataset:"
echo "   sbatch run_ukbb_inference.sh"
echo ""
echo "All tools are pre-downloaded and ready to use!"
echo "=========================================="
