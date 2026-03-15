#! /bin/bash

SMILES=data/caspirus10k.smi
OUTPUT=caspyrus10k
CONFIG=configs/medusa_default_config.yml
PROCESSES=0
STOCK=paroutes

CUDA_VISIBLE_DEVICES=0 aizynthcli --config $CONFIG \
--smiles $SMILES \
--output ${OUTPUT}_${STOCK}_medusa.json \
--stocks $STOCK \
--nproc $PROCESSES \
 --log_to_file
