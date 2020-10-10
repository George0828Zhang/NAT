#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL

raw=$RAW/iwslt/en-de
ready=$CACHE/ready

mkdir -p $ready

for L in $src $tgt; do
    if [[ ! -z "$distill" ]]; then
        disdir=$src$tgt # ende or deen
        cp $raw/distill/$disdir/train.tags.en-de.bpe.$L $ready/train.$L
    else
        cp $raw/train/train.tags.en-de.bpe.$L $ready/train.$L
    fi
    cp $raw/dev/valid.bpe.$L $ready/valid.$L
done

