#!/bin/bash
# LADA official: 16-shot, seed=$1, gpu=$2
set -e
SEED=$1; GPU=$2; DIR="LADA_official_s${SEED}"

OPTS="num_shots 16 root /data1/open_datasets/X-TAIL seed $SEED gpu $GPU output_dir $DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset aircraft num_epochs 40 continue_train_first True
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset caltech101 num_epochs 10
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset dtd num_epochs 30
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset eurosat num_epochs 100
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset flowers num_epochs 30
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset food101 num_epochs 5
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset mnist num_epochs 200
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset oxford_pets num_epochs 10
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset stanford_cars num_epochs 30
python3 -u main.py -d TAIL -m clip_vit_b16 $OPTS dataset sun397 num_epochs 10

python3 result_process.py -d TAIL --output_dir $DIR
echo "=== LADA seed $SEED DONE ==="