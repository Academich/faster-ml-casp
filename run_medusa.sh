#! /bin/bash

SMILES=caspyrus_10k.smi
OUTPUT=caspyrus_10k
CONFIG=configs/transformer_medusa_default_config.yml
PROCESSES=0

CUDA_VISIBLE_DEVICES=0 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_medusa.json \
--nproc $PROCESSES \
 --log_to_file
