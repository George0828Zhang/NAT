#!/usr/bin/env bash
# setup cache dir for raw, 
export RAW=$PREFIX/iwslt16
if [[ ! -z "$DISTILL" ]]; then
    export DATASET="iwslt16.en-de.xlmr.distill"
else
    export DATASET="iwslt16.en-de.xlmr"
fi
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET

xlmrloc=/home/george/Projects/NAT/bert2nat/xlmr
export SPM_MODEL=$xlmrloc/sentencepiece.bpe.model
export BINARIZEARGS="--joined-dictionary --bpe sentencepiece --srcdict $xlmrloc/dict.txt"