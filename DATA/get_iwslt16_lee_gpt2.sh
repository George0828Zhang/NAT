#!/usr/bin/env bash
#
WORKERS=8
BIN=$(pwd)/data-bin
TMP=/media/george/Data/iwslt16.lee.raw
OUTDIR=$BIN/iwslt16.gpt2.en-de
EXTRAOPTIONS=$1

if [[ $EXTRAOPTIONS = 'distill' ]]; then
    OUTDIR=$OUTDIR.distill
fi


URL="https://drive.google.com/u/0/uc?id=1m7dZqEXHWPYcre6xxsFwFLrb9CRCZGmn&export=download"
GZ=iwslt.tar.gz

src=en
tgt=de
lang=en-de

mkdir -p $TMP

echo "Downloading data from ${URL}..."
cd $TMP
if [ -f $GZ ]; then
    echo "Data already downloaded."
else
    gdown "$URL"
fi

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ

for L in $src $tgt; do
    prep=$TMP/iwslt/en-de
    if [[ $EXTRAOPTIONS = 'distill' ]]; then
        cp $prep/distill/ende/train.tags.en-de.bpe.$L $TMP/train.$L
    else
        cp $prep/train/train.tags.en-de.bpe.$L $TMP/train.$L
    fi
    cp $prep/dev/valid.bpe.$L $TMP/valid.$L
done

# GPT2 BPE encode the source side
cd $TMP
if [[ ! -d gpt2_bpe ]]; then
    mkdir -p gpt2_bpe
    wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fi

for SPLIT in train valid; do
    cat $TMP/$SPLIT.$src | sed 's/@@ //g' > $TMP/$SPLIT.$src.raw

    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs $TMP/$SPLIT.$src.raw \
        --outputs $TMP/$SPLIT.$src \
        --keep-empty \
        --workers $WORKERS
done

mkdir -p $OUTDIR
TEXT=$TMP
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $OUTDIR \
    --workers $WORKERS

