#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready

mkdir -p $prep $ready

for L in $src $tgt; do
    if [[ ! -z "$DISTILL" ]]; then
        if [[ $tgt == "en" ]]; then
            echo "Distillation $src->$tgt is not done on this dataset."
            exit
        fi

        cat $raw/wmt14_ende_distill/train.en-de.$L \
            | sed 's/@@ //g' > $prep/train.$L
    else        
        cat $raw/wmt14_ende/train.en-de.$L \
            | sed 's/@@ //g' > $prep/train.$L
    fi

    cat $raw/wmt14_ende/valid.en-de.$L \
        | sed 's/@@ //g' > $prep/valid.$L
    cat $raw/wmt14_ende/test.en-de.$L \
        | sed 's/@@ //g' > $prep/test.$L
done

