#!/usr/bin/env bash
#
WORKERS=8
BIN=$(pwd)/data-bin
TMP=/media/george/Data/iwslt16.lee.raw
OUTDIR=$BIN/iwslt16.distill.en-de

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
cd ..



for L in $src $tgt; do
    prep=$TMP/iwslt/en-de
    cp $prep/distill/ende/train.tags.en-de.bpe.$L $TMP/train.$L
    cp $prep/dev/valid.bpe.$L $TMP/valid.$L
done

mkdir -p $OUTDIR
TEXT=$TMP
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $OUTDIR \
    --workers $WORKERS

