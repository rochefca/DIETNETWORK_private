# DietNetwork Model Upload Instructions

## Summary of Changes

I've implemented a complete smoke test system and model preset loading for DietNetwork. Here's what was created:

### New Files Created:
1. **`Dietnet/pretrained_models.py`** - Model registry with preset definitions
2. **`Dietnet/model_manager.py`** - Download, cache, and validation logic
3. **`scripts/package_models.sh`** - Create model tarball for upload
4. **`scripts/package_test_data.sh`** - Create test data tarball for upload
5. **`tests/smoke_test_config.sh`** - Test configuration
6. **`tests/download_test_data.sh`** - Download test data from URL
7. **`tests/run_smoke_test.sh`** - Main smoke test orchestrator
8. **`tests/data/.gitignore`** - Ignore downloaded data

### Modified Files:
1. **`Dietnet/cli.py`** - Updated `predict` command with:
   - New `--model` option (accepts presets or local paths)
   - Deprecation warnings for HDF5 approach
   - Backward compatibility maintained
2. **`Dietnet/predict_with_plink.py`** - Added ensemble inference:
   - `predict_ensemble()` function for multi-model inference
   - Majority voting across models
   - Model agreement statistics in output
3. **`.gitignore`** - Added cache and test data exclusions
4. **`README.md`** - Added smoke test documentation

---

## What You Need to Upload

### Step 1: Package the Files

Run these scripts on the login node (they have access to the data):

```bash
# Package the pretrained models (pretrained_1000g_models/ - 15 models ensemble)
bash scripts/package_models.sh
# Output: dietnet_1kgp_default_v1.tar.gz (~1.6 GB) + SHA256 checksum

# Package the test data
bash scripts/package_test_data.sh
# Output: dietnet_1kgp_test_data_v1.tar.gz + SHA256 checksum
```

### Step 2: Upload to Server

Upload both `.tar.gz` files to:
- Zenodo (https://zenodo.org/) - Recommended for academic data
- Figshare (https://figshare.com/)
- Institutional server
- GitHub Releases (if repository is public and files are small enough)

**Files to upload:**
1. `dietnet_1kgp_default_v1.tar.gz` (~1.6 GB) - Model package (15 models: 3 seeds × 5 folds)
2. `dietnet_1kgp_test_data_v1.tar.gz` (~1-2 GB) - Test data

### Step 3: Update Configuration

After upload, you'll get download URLs. Update `Dietnet/pretrained_models.py`:

```python
PRETRAINED_MODELS = {
    "1kgp_default": {
        ...
        "model_url": "https://YOUR_SERVER/dietnet_1kgp_default_v1.tar.gz",  # <-- Add URL here
        "model_sha256": "abc123...",  # <-- Add SHA256 from packaging script
        ...
    }
}

TEST_DATA = {
    "1kgp_default": {
        ...
        "plink_url": "https://YOUR_SERVER/dietnet_1kgp_test_data_v1.tar.gz",  # <-- Add URL here
        "plink_sha256": "def456...",  # <-- Add SHA256 from packaging script
        ...
    }
}
```

---

## How Users Will Use This

### Running the Smoke Test

```bash
# Download, install, and test everything automatically
bash tests/run_smoke_test.sh
```

This will:
1. Download the model preset (if not cached) → `~/.cache/dietnet/1kgp_default_v1/` (15 models: 3 seeds × 5 folds)
2. Download the test data (if not exists) → `tests/data/1kgp/`
3. Run ensemble inference on the test data with all 15 models
4. Compute majority vote predictions with model agreement statistics
5. Validate accuracy is 85-100%
6. Report PASS/FAIL

### Using Model Presets in CLI

```bash
# Option 1: Use preset (downloads automatically, runs ensemble with all models)
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/user/data \
                --output predictions.tsv

# Option 2: Use local model package (runs ensemble if multiple models present)
dietnet predict --model ./pretrained_1000g_models \
                --plink-prefix /path/to/user/data \
                --output predictions.tsv
```

**Note:** Ensemble inference is automatic when the model package contains multiple seeds/folds. Output includes `model_agreement` column showing prediction confidence.

### Legacy HDF5 Approach (Deprecated)

Old command still works but shows deprecation warning:
```bash
dietnet predict --model-params best_model.pt \
                --test-dataset test.hdf5 \
                --train-dataset train.hdf5 \
                --config config.yaml \
                --embedding embedding.npz \
                --input-features-stats input_stats.npz \
                --output-dir ./results \
                --which-fold 0
```

---

## Adding Future Model Presets (e.g., hgdp_ukbb)

When you train the HGDP+1KGP model:

1. **Package it:**
   ```bash
   # Update package_models.sh to use your new model directory
   # Then run:
   bash scripts/package_models.sh
   ```

2. **Upload the tarball** and get URL + SHA256

3. **Update the registry** in `Dietnet/pretrained_models.py`:
   ```python
   "hgdp_ukbb": {
       "name": "HGDP + 1KGP for UKBB Inference",
       "version": "v1",
       "description": "Trained on HGDP+1KGP for UKBB population inference",
       "model_url": "https://YOUR_SERVER/dietnet_hgdp_ukbb_v1.tar.gz",
       "model_sha256": "sha256_here",
       "cache_dir": "hgdp_ukbb_v1",
       "n_classes": XX,  # Update with actual value
       "n_snps": XXXXX,  # Update with actual value
   }
   ```

4. **Users can immediately use it:**
   ```bash
   dietnet predict --model hgdp_ukbb \
                   --plink-prefix ukbb_data \
                   --output predictions.tsv
   ```

---

## File Structure After Implementation

```
DIETNETWORK/
├── Dietnet/
│   ├── pretrained_models.py      # NEW: Model registry
│   ├── model_manager.py          # NEW: Download/cache logic
│   ├── cli.py                    # MODIFIED: Updated predict command
│   └── predict_with_plink.py     # MODIFIED: Added ensemble inference
├── scripts/                       # NEW: Packaging scripts
│   ├── package_models.sh
│   └── package_test_data.sh
├── tests/
│   ├── smoke_test_config.sh      # NEW
│   ├── download_test_data.sh     # NEW
│   ├── run_smoke_test.sh         # NEW: Main smoke test
│   ├── check_predictions.py      # Existing
│   └── data/                     # NEW: Downloaded data (git-ignored)
│       └── .gitignore
├── pretrained_1000g_models/      # Existing model package (15 models)
├── .gitignore                    # MODIFIED: Added cache exclusions
├── README.md                     # MODIFIED: Added docs
└── UPLOAD_INSTRUCTIONS.md        # NEW: This file
```

---

## Testing Before Upload

You can test the packaging scripts now:

```bash
# Test model packaging (creates tarball)
bash scripts/package_models.sh

# Test data packaging (creates tarball, requires access to shared data)
bash scripts/package_test_data.sh
```

This will show you:
- File sizes
- SHA256 checksums
- What to upload

---

## Benefits of This System

1. **Reproducible testing** - Anyone can validate their installation
2. **Easy distribution** - Users don't need access to your HPC cluster
3. **Automatic caching** - Models downloaded once, reused forever
4. **Ensemble inference** - Automatically runs all 15 models and provides agreement statistics
5. **Extensible** - Adding new presets is trivial
6. **Backward compatible** - Old HDF5 approach still works (with deprecation warning)
7. **Self-documenting** - Smoke test shows users how the system works

---

## Questions?

If you have questions about:
- Where to upload (Zenodo recommended for academic data)
- How to update URLs after upload
- Adding new model presets
- Troubleshooting the smoke test

Let me know and I can help!
