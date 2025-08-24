# Multi-step retrosynthesis with the SMILES-to-SMILES transformer accelerated by speculative beam search

## Overview
Multi-step CASP with AiZynthFinder including a single-step SMILES-to-SMILES transformer accelerated by speculative decoding.
Uses the code of [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) and [AiZynthFinder model zoo](https://github.com/PTorrenPeraire/modelsmatter_modelzoo) with some modifications.
The single-step transformer accelerated with Medusa is implemented in [its own repository](https://github.com/Academich/SMILES-to-SMILES-transformer).

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

## Checkpoints
Link to the single-step models:
1. Basic transformer: https://drive.google.com/drive/folders/1v4pKYWlE0qNA-ksa7yX55i7qMeesURON
2. Medusa: https://drive.google.com/drive/folders/1uE8J13AgPpfLRJuGuXBOJS1aP2IA1fTk?usp=sharing

## Building blocks
The set of building blocks and the Caspyrus10k dataset used in this work can be found in this [Figshare repository](https://figshare.com/s/2eab4132b322229c1efc).

## References
1. Genheden, S., Thakkar, A., Chadimová, V., Reymond, J., Engkvist, O., & Bjerrum, E. (2020). AiZynthFinder (Version 2.2.1) [Computer software]. https://doi.org/https://doi.org/10.1186/s13321-020-00472-1
2. Torren-Peraire, P., Hassen, A., Genheden, S., Verhoeven, J., Clevert, D., Preuss, M., & Tetko, I. (2022). Models Matter: the impact of single-step retrosynthesis on synthesis planning. https://doi.org/10.1039/D3DD00252G
