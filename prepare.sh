#!/usr/bin/env bash
ROOT=../
WANDB=$(pwd)/wandb_inject.txt
cd $ROOT
git clone https://github.com/pytorch/fairseq
cd fairseq
if grep -Fq -f $WANDB fairseq_cli/train.py;then
    echo "injection already done!"
else
    sed -i "/def main\(.*\)\:/r$WANDB" fairseq_cli/train.py
fi
pip install --user .