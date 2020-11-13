#!/usr/bin/env bash
# setup cache dir for raw,
export RAW=$PREFIX/iwslt14.enes
export DATASET="iwslt14.enes"
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

export SPM_MODEL="Current"
export BINARIZEARGS="--joined-dictionary --bpe sentencepiece"