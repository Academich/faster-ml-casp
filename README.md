# Multi-step retrosynthesis with the SMILES-to-SMILES transformer accelerated by speculative beam search

## Overview
Multi-step CASP with AiZynthFinder including a single-step SMILES-to-SMILES transformer accelerated by speculative decoding.
Uses the code of [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) and [AiZynthFinder model zoo](https://github.com/PTorrenPeraire/modelsmatter_modelzoo) with some modifications.
The single-step transformer accelerated with Medusa is implemented in [its own repository](https://doi.org/10.5281/zenodo.18002214).

## Installation

1. Install `uv` (if not already installed). It is a faster alternative to `pip` and `poetry`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install the virtual environment using `uv sync`. Different installation is possible for different CUDA versions.

```bash
uv sync --extra cpu      # CPU-only
uv sync --extra cu118    # CUDA 11.8
uv sync --extra cu124    # CUDA 12.4
uv sync --extra cu128    # CUDA 12.8
uv sync <other args> --no-dev  # Exclude dev dependencies
```

3. Activate the virtual environment.

```bash
source .venv/bin/activate
```

4. For development, install pre-commit hooks.

```bash
pre-commit install
```

## Checkpoints and data

### Template-free single-step models

Links to the single-step models:
1. Basic transformer: https://drive.google.com/drive/folders/1v4pKYWlE0qNA-ksa7yX55i7qMeesURON
2. Medusa: https://drive.google.com/drive/folders/1uE8J13AgPpfLRJuGuXBOJS1aP2IA1fTk

Download the trained checkpoints of the single-step models and the corresponding vocabs using `gdown`:
```bash
mkdir checkpoints

# Transformer
gdown https://drive.google.com/drive/folders/1v4pKYWlE0qNA-ksa7yX55i7qMeesURON -O checkpoints/retro/transformer --folder

# Medusa
gdown https://drive.google.com/drive/folders/1uE8J13AgPpfLRJuGuXBOJS1aP2IA1fTk -O checkpoints/retro/medusa --folder
```

### Template-based AiZynthFinder model, building blocks, etc.

Run the following command:
```bash
download_public_data ./aizynthfinder/public
```

This is an alias for `aizynthfinder/tools/download_public_data.py`

It downloads:
  - Template-based single-step retrosynthesis model from AiZynthFinder: `.onnx` files, uspto templates
  - Building blocks: `zinc_stock.hdf5` (17,422,831 molecules) and `paroutes_n1_stock.hdf5` (13,141 molecules)
  - Caspyrus10k dataset: `caspyrus10k.csv` (10,000 molecules) turned into `caspirus10k.smi`
  - PaRoutes dataset: `n1-routes.json`, `n1-targets.txt`

It also generates the default config for AiZynthFinder: config.yaml
All the files are saved to the directory `aizynthfinder/public`

## Building blocks
For reference, the building block stocks can be also found here:
ZINC: https://doi.org/10.6084/m9.figshare.12334577.v1
PaRoutes (and Caspyrus10k): https://figshare.com/s/2eab4132b322229c1efc


## References
1. Genheden, S., Thakkar, A., Chadimová, V., Reymond, J., Engkvist, O., & Bjerrum, E. (2020). AiZynthFinder (Version 2.2.1) [Computer software]. https://doi.org/https://doi.org/10.1186/s13321-020-00472-1
2. Torren-Peraire, P., Hassen, A., Genheden, S., Verhoeven, J., Clevert, D., Preuss, M., & Tetko, I. (2022). Models Matter: the impact of single-step retrosynthesis on synthesis planning. https://doi.org/10.1039/D3DD00252G
