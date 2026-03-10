#! /bin/bash

SMILES=data/n1-targets.txt
OUTPUT=paroutes_n1
CONFIG=configs/transformer_paroutes_n1_config.yml
PROCESSES=0

CUDA_VISIBLE_DEVICES=0 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_transformer.json \
--nproc $PROCESSES \
--log_to_file
