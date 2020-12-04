#!/usr/bin/env bash
# setup cache dir for raw,
export RAW=$PREFIX/wmt14
if [[ ! -z "$DISTILL" ]]; then
    export DATASET="wmt14.en-de.distill"
else
    export DATASET="wmt14.en-de"
fi
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

export SPM_MODEL='Current'
export BINARIZEARGS="--joined-dictionary --bpe sentencepiece"