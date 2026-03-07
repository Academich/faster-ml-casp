# Multi-step retrosynthesis with the SMILES-to-SMILES transformer accelerated by speculative beam search

## Overview
Multi-step CASP with AiZynthFinder including a single-step SMILES-to-SMILES transformer accelerated by speculative decoding.
Uses the code of [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) and [AiZynthFinder model zoo](https://github.com/PTorrenPeraire/modelsmatter_modelzoo) with some modifications.
The single-step transformer accelerated with Medusa is implemented in [its own repository](https://doi.org/10.5281/zenodo.18002214).

## Installation
Install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/).
Create the new environment with `uv sync` and activate it:

For production:
```bash
uv sync --no-group dev
source .venv/bin/activate
```

For development, install the pre-commit hooks:
```bash
uv sync
source .venv/bin/activate
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

### Template-based AiZynthFinder model and ZINC stock

To download the default AiZynthFinder template-based single-step model:
```bash
download_public_data ./aizynthfinder/public
``` 

## Building blocks
The set of building blocks ("paroutes_stock.hdf5" containing 13414 molecules - PaRoutes stock-n1 ) and the Caspyrus10k ("CASPyrus10k.csv") dataset used in this work can be found in this [Figshare repository](https://figshare.com/s/2eab4132b322229c1efc).
ZINC stock (17,422,831 molecules) "zinc_stock_17_04_20.hdf5" can be found here https://doi.org/10.6084/m9.figshare.12334577.v1
AiZynthFinder needs only the SMILES from CASPyrus10k.csv:
```bash
import pandas

pandas.read_csv('CASPyrus10k.csv')["smiles"].to_csv('CASPyrus10k.smi', header=False, index=False)
```

## References
1. Genheden, S., Thakkar, A., Chadimová, V., Reymond, J., Engkvist, O., & Bjerrum, E. (2020). AiZynthFinder (Version 2.2.1) [Computer software]. https://doi.org/https://doi.org/10.1186/s13321-020-00472-1
2. Torren-Peraire, P., Hassen, A., Genheden, S., Verhoeven, J., Clevert, D., Preuss, M., & Tetko, I. (2022). Models Matter: the impact of single-step retrosynthesis on synthesis planning. https://doi.org/10.1039/D3DD00252G
