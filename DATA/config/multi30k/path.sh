#!/usr/bin/env bash
# setup cache dir for raw, 
export RAW=$PREFIX/multi30k
src=$SRCLANG
tgt=$TGTLANG
if [[ ! -z "$DISTILL" ]]; then
    export DATASET="multi30k.$src-$tgt.distill"
else
    export DATASET="multi30k.$src-$tgt"
fi
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

export BPE_CODE='Current'
export BINARIZEARGS="--joined-dictionary --bpe subword_nmt"