#!/usr/bin/env bash
# setup cache dir for raw, 
export RAW=$PREFIX/iwslt16
if [[ ! -z "$DISTILL" ]]; then
    export DATASET="iwslt16.en-de.distill"
else
    export DATASET="iwslt16.en-de"
fi
export CACHE=$PREFIX/$DATASET
export OUTDIR=$DATABIN/$DATASET
export BINARIZEARGS="--joined-dictionary"