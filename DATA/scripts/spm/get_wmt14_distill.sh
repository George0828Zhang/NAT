#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
PREFIX=${1:='.'}

WORKERS=4

URLS=(
    "http://dl.fbaipublicfiles.com/nat/distill_dataset.zip"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "distill_dataset.zip"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "wmt14_ende_distill/train.en-de"
)

OUTDIR=wmt14_distill

TMP=$PREFIX/$OUTDIR

src=en
tgt=de
lang=en-de
tmp=$TMP/tmp
# prep=$TMP/prep
orig=$TMP/orig
mkdir -p $tmp $orig $prep

cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done

cd $TMP

echo "pre-processing train data..."
for l in $src $tgt; do
    tok=$tmp/train.$l
    if [ -f $tok ]; then
        echo "found $tok, skipping detok."
    else        
        for f in "${CORPORA[@]}"; do
            echo "detokenizing $orig/$f.$l to $tok"
            cat $orig/$f.$l | \
            sed -r 's/(@@ )|(@@ ?$)//g' | \
            sacremoses -l $l -j $WORKERS detokenize >> $tok
        done
    fi
done

echo "pre-processing valid data..."
for l in $src $tgt; do
    tok=$tmp/valid.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2013-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $tok
    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    tok=$tmp/test.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $tok
    echo ""
done