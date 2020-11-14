#!/usr/bin/env bash
# setup cache dir for raw,
export RAW=$PREFIX/wmt17
export DATASET="wmt17.zh-en"
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

export SPM_MODEL="Current"
export BINARIZEARGS="--joined-dictionary --bpe sentencepiece"