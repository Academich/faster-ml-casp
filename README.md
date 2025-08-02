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
uv pre-commit install
```

## Data
**Download USPTO 50K** and augment it using [RSMILES](https://github.com/otori-bird/retrosynthesis) augmentation.  
Clone the RSMILES repository to some path in your system.
```bash
THIS_REPO_PATH=$(pwd) # The full path to this repository 
RSMILES_PATH=../retrosynthesis  # as an example; the path to the RSMILES repository

gdown https://drive.google.com/drive/folders/1la4OgBKgm2K-IRwuV-GHUNjN3bcCrl6v -O ${RSMILES_PATH}/dataset/USPTO_50K --folder
cd ${RSMILES_PATH}
AUGMENTATIONS=20
PROCESSES=8
python3 preprocessing/generate_PtoR_data.py -augmentation ${AUGMENTATIONS} -processes ${PROCESSES} -test_except
python3 preprocessing/generate_PtoR_data.py -augmentation 1 -processes ${PROCESSES} -test_only -canonical
mv dataset/USPTO_50K_PtoR_aug${AUGMENTATIONS} ${THIS_REPO_PATH}/data # The augmented dataset is now in this repository
mv dataset/USPTO_50K_PtoR_aug1 ${THIS_REPO_PATH}/data
cd $THIS_REPO_PATH
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug1/test
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug${AUGMENTATIONS}/train
python3 src/detokenize.py --data_dir data/USPTO_50K_PtoR_aug${AUGMENTATIONS}/val
```

## Checkpoints
Link to the single-step models:
1. Basic transformer: https://drive.google.com/drive/folders/1v4pKYWlE0qNA-ksa7yX55i7qMeesURON
2. Medusa: https://drive.google.com/file/d/1jZfD0Fqj2P5Is5fK_9inUepM5rfe4AGM/view?usp=drive_link

## Building blocks
The set of building blocks and the Caspyrus10k dataset used in this work can be found in this [Figshare repository](https://figshare.com/s/2eab4132b322229c1efc).

## References
1. Genheden, S., Thakkar, A., Chadimová, V., Reymond, J., Engkvist, O., & Bjerrum, E. (2020). AiZynthFinder (Version 2.2.1) [Computer software]. https://doi.org/https://doi.org/10.1186/s13321-020-00472-1
2. Torren-Peraire, P., Hassen, A., Genheden, S., Verhoeven, J., Clevert, D., Preuss, M., & Tetko, I. (2022). Models Matter: the impact of single-step retrosynthesis on synthesis planning. https://doi.org/10.1039/D3DD00252G
