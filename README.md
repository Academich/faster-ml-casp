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
download_public_data data
```

This is an alias for `aizynthfinder/tools/download_public_data.py`

It downloads:
  - Template-based single-step retrosynthesis model from AiZynthFinder: `.onnx` files, uspto templates
  - Building blocks: `zinc_stock.hdf5` (17,422,831 molecules) and `paroutes_n1_stock.hdf5` (13,141 molecules)
  - Caspyrus10k dataset: `caspyrus10k.csv` (10,000 molecules) turned into `caspirus10k.smi`
  - PaRoutes dataset: `n1-routes.json`, `n1-targets.txt`

It also generates the default config for AiZynthFinder: config.yaml
All the files are saved to the directory `data`

### Downsampling PaRoutes n1-targets
For the experiment with a 1k subsample of n1-targets, the subsample has been obtained as follows:
```python
import random

filename = "data/n1-targets.txt"
with open(filename, "r") as fileobj:        
    smiles = [line.strip() for line in fileobj.readlines()]

rng = random.Random(42)
rng.shuffle(smiles)
smiles_random = smiles[:1000]

text = "\n".join(smiles_random) + "\n"
with open("data/n1-targets_1000_subsample.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

## USPTO 50K
We use the same version of USPTO 50K as in the GLN paper (https://github.com/Hanjun-Dai/GLN) with splits originally produced
using a script from the retrosym paper (https://github.com/connorcoley/retrosim/blob/master/retrosim/data/get_data.py).
The files are provided in `data/uspto50k_test`.

## Building blocks
For reference, the building block stocks can be also found here:
ZINC: https://doi.org/10.6084/m9.figshare.12334577.v1
PaRoutes (and Caspyrus10k): https://figshare.com/s/2eab4132b322229c1efc

## Usage
For synthesis planning with Transformer, run:

```bash
bash run_transformer.sh
```

For synthesis planning with Medusa, run:

```bash
bash run_medusa.sh
```
To select the building block stock, use either `STOCK=paroutes` or `STOCK=zinc` in the bash scripts.

To run the experiments on PaRoutes-n1, run
```bash
bash run_transformer_paroutes_n1.sh
```
or

```bash
bash run_medusa_paroutes_n1.sh
```
### Configuration
The configuration files for AiZynthFinder, Transformer or Medusa can be found in `configs/`.

AiZynthFinder allows constraining the planning by both total time and the maximum number of iterations, and these must be set simultaneously. 
Therefore, those constraints should be chosen carefully, because ultimately, the strictest of the two is the limiting constraint. 
In our experiments, we chose either a time limit (assigning a very large maximum number of iterations) or an iteration limit (assigning a very large maximum time). 
This made it easier to analyze the experiments.
To change the time limit for synthesis planning, change the number of seconds in `search` -> `time_limit` (e.g., `5`, `15` or `180`). 
To change the number of iterations of planning, change `search` -> `iteration_limit`.  

To change beam size of the single step model, change `expansion` -> `data` -> `kwargs` -> `beam_size` (e.g., `10` or `50`). 
To change the planning algorithm, change `search` -> `algorithm` (e.g., `aizynthfinder.search.retrostar.search_tree.SearchTree` or `aizynthfinder.search.dfpn.search_tree.SearchTree`).
To change maximal depth of route, change `search` -> `max_transforms` (e.g., `5` or `7`).  
`search` -> `return_first: true` means that the planning is stopped if the first complete route is found. 


### To measure round-trip accuracy for Transformer or Medusa you need to download the transformer product model
# Transformer product model
mkdir checkpoints/reaction_prediction
gdown https://drive.google.com/drive/folders/1sBiVgFZyD4F42nVqR835-0Tl90LkQvU9 -O checkpoints/reaction_prediction --folder

To change beam size of the single step model at configs/medusa_default_config.yml or configs/transformer_default_config.yml, 
change `expansion` -> `data` -> `kwargs` -> `beam_size` (e.g., `10` or `50`). 

To evaluate Medusa do:
```bash
CUDA_VISIBLE_DEVICES=0 python3 round_trip_transformer_script.py --use_gpu --retro_model_config configs/medusa_default_config.yml
```

To evaluate Transformer single-step retrosynthesis model:
```bash
CUDA_VISIBLE_DEVICES=0 python3 round_trip_transformer_script.py --use_gpu --retro_model_config configs/transformer_default_config.yml
```
## References
1. Genheden, S., Thakkar, A., Chadimová, V., Reymond, J., Engkvist, O., & Bjerrum, E. (2020). AiZynthFinder (Version 2.2.1) [Computer software]. https://doi.org/https://doi.org/10.1186/s13321-020-00472-1
2. Torren-Peraire, P., Hassen, A., Genheden, S., Verhoeven, J., Clevert, D., Preuss, M., & Tetko, I. (2022). Models Matter: the impact of single-step retrosynthesis on synthesis planning. https://doi.org/10.1039/D3DD00252G
