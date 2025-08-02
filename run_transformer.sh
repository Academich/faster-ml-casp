#! /bin/bash

SMILES=caspyrus_10k.smi
OUTPUT=caspyrus_10k
CONFIG=configs/transformer_transformer_default_config.yml
PROCESSES=0

CUDA_VISIBLE_DEVICES=4 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_transformer.json \
--nproc $PROCESSES \
 --log_to_file
