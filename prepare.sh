#!/usr/bin/env bash
ROOT=../
WANDB=$(pwd)/wandb_inject.py
cd $ROOT
git clone https://github.com/pytorch/fairseq
cd fairseq
python $WANDB fairseq_cli/train.py
pip install --user --upgrade . 

