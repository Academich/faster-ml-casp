#! /bin/bash

SMILES=data/n1-targets.txt  # or data/n1-targets_1000_subsample.txt
OUTPUT=paroutes_n1
CONFIG=configs/medusa_paroutes_n1_config.yml
PROCESSES=0

CUDA_VISIBLE_DEVICES=0 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_medusa.json \
--nproc $PROCESSES \
--log_to_file
