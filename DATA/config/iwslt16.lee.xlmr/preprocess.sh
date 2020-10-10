#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL

raw=$RAW/iwslt/en-de
prep=$CACHE/prep
ready=$CACHE/ready

mkdir -p $prep $ready

for L in $src $tgt; do
    if [[ ! -z "$distill" ]]; then
        disdir=$src$tgt # ende or deen
        cat $raw/distill/$disdir/train.tags.en-de.bpe.$L \
        | sed 's/@@ //g' > $prep/train.$L
    else
        cat $raw/train/train.tags.en-de.bpe.$L \
        | sed 's/@@ //g' > $prep/train.$L
    fi
    cat $raw/dev/valid.bpe.$L \
    | sed 's/@@ //g' > $prep/valid.$L
done

