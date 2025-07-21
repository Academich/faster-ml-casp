#! /bin/bash

SMILES=/home/mikhail/work/faster_ml_casp/caspyrus_10_tmp.smi
OUTPUT=/home/mikhail/work/faster_ml_casp/caspyrus_10tmp
CONFIG=/home/mikhail/work/faster_ml_casp/configs/transformer_transformer_default_config.yml
PROCESSES=0

CUDA_VISIBLE_DEVICES=4 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_transformer.json \
--nproc $PROCESSES \
# --log_to_file
