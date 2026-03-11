# Upload Instructions

Reference guide for packaging and distributing pretrained models and test data.

## Step 1: Package Models

```bash
# Default: packages pretrained_1000g_models/ → dietnet_1kgp_default_v1.tar.gz
bash scripts/package_models.sh

# Custom source dir and output name
bash scripts/package_models.sh my_model_dir my_model_v1.tar.gz
```

The script expects the source directory to contain `seed_*/` subdirectories (one per seed×fold combination). It will warn if the count is not 15 (3 seeds × 5 folds).

## Step 2: Package Test Data

```bash
# Set source paths via environment variables, then run:
PLINK_PREFIX=/path/to/plink_prefix \
LABELS=/path/to/labels.tsv \
OUTPUT=dietnet_1kgp_test_data_v1.tar.gz \
bash scripts/package_test_data.sh
```

Or edit `PLINK_PREFIX`, `LABELS`, and `OUTPUT` directly at the top of the script.

## Step 3: Upload Tarballs

Upload the generated `.tar.gz` files to a public host. Recommended:
- [Zenodo](https://zenodo.org/) — DOI-minted, free for academic data
- [Figshare](https://figshare.com/)
- Institutional server or GitHub Releases (for smaller files)

## Step 4: Update `Dietnet/pretrained_models.py`

After upload, copy the download URLs and SHA256 checksums (printed by the packaging scripts) into `Dietnet/pretrained_models.py`:

```python
PRETRAINED_MODELS = {
    "1kgp_default": {
        ...
        "model_url": "https://zenodo.org/record/XXXXX/files/dietnet_1kgp_default_v1.tar.gz",
        "model_sha256": "<sha256 from package_models.sh>",
        ...
    }
}

TEST_DATA = {
    "1kgp_default": {
        ...
        "plink_url": "https://zenodo.org/record/XXXXX/files/dietnet_1kgp_test_data_v1.tar.gz",
        "plink_sha256": "<sha256 from package_test_data.sh>",
        ...
    }
}
```

## Adding Future Presets (e.g., `hgdp_ukbb`)

1. Package with `bash scripts/package_models.sh hgdp_ukbb_models dietnet_hgdp_ukbb_v1.tar.gz`
2. Upload and record URL + SHA256
3. Add an entry to `Dietnet/pretrained_models.py`:
   ```python
   "hgdp_ukbb": {
       "name": "HGDP + 1KGP for UKBB Inference",
       "version": "v1",
       "model_url": "https://...",
       "model_sha256": "...",
       "cache_dir": "hgdp_ukbb_v1",
   }
   ```
4. Users can immediately use: `dietnet predict --model hgdp_ukbb ...`

## File Structure

```
DIETNETWORK/
├── Dietnet/
│   ├── pretrained_models.py      # Model registry (URLs + SHA256)
│   ├── model_manager.py          # Download/cache logic
│   └── cli.py
├── scripts/
│   ├── package_models.sh         # Packages seed_*/ → tarball
│   ├── package_test_data.sh      # Packages PLINK + labels → tarball
│   ├── train_predict.sh          # Train + predict template
│   └── predict_external.sh       # Inference-only template
├── tests/
│   └── kgp_precomputed/
│       ├── run_smoke_test.sh     # Ensemble inference smoke test
│       ├── run_smoke_test_single.sh
│       ├── run_train_smoke.sh    # Full train+predict+check smoke test
│       ├── download_test_data.sh
│       ├── download_model.sh
│       └── data/                 # Downloaded data (git-ignored)
└── untracked_scripts/            # Site-specific SLURM scripts (git-ignored)
```
