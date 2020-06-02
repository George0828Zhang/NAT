#!/usr/bin/env bash
ROOT=../
WANDB=$(pwd)/wandb_inject.py
cd $ROOT
if [ -d fairseq ]; then
    cd fairseq
    git pull
    cd ..
else
    git clone https://github.com/pytorch/fairseq
fi
cd fairseq
# python $WANDB fairseq_cli/train.py
pip install --user --upgrade . 

