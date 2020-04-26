#!/usr/bin/env bash

ROOT_DIR=$1 # dataset/tmp
SRC=$2 # de
TGT=$3 # en
BPE_TOKENS=$4 # 40000
BPEROOT=subword-nmt/subword_nmt

TRAIN=$ROOT_DIR/all.tok
BPE_CODE=$ROOT_DIR/code
rm -f $TRAIN $BPE_CODE

if [ -f $TRAIN ]; then
    echo "${TRAIN} found, skipping concat."
else
    for SPLIT in train valid; do \
        for l in $SRC $TGT; do \
            cat $ROOT_DIR/${SPLIT}.${l} >> $TRAIN
        done
    done
fi

echo "learn_bpe.py on ${TRAIN}..."
if [ -f $BPE_CODE ]; then
    echo "${BPE_CODE} found, skipping learn."
else
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
fi
