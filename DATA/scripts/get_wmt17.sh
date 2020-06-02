#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
PREFIX=${1:='.'}
ICML17=$2
WORKERS=4
TMP=$PREFIX/wmt_en_de
mkdir -p $TMP

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
cd $TMP

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ $ICML17 == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
    OUTDIR=wmt14_en_de
else
    OUTDIR=wmt17_en_de
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
tmp=$TMP/tmp
prep=$TMP/prep
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
        fi
    fi
done

cd $TMP

echo "pre-processing train data..."
for l in $src $tgt; do
    tok=$prep/train.$l
    if [ -f $tok ]; then
        echo "found $tok, skipping tokenization."        
    else
        for f in "${CORPORA[@]}"; do
            # cat $orig/$f.$l | \
            #     perl $NORM_PUNC $l | \
            #     perl $REM_NON_PRINT_CHAR | \
            #     perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l 
            # # -a : aggressive hyphen splitting
            cat $orig/$f.$l | \
            $REPLACE_UNICODE_PUNCT | \
                $NORM_PUNC -l $l | \
                $REM_NON_PRINT_CHAR | \
                $TOKENIZER -l $l -no-escape -a -threads $WORKERS | \
                $LC >> $tok
        done
    fi
done

echo "pre-processing valid data..."
for l in $src $tgt; do
    tok=$prep/valid.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2013-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        $REPLACE_UNICODE_PUNCT | \
            $NORM_PUNC -l $l | \
            $REM_NON_PRINT_CHAR | \
            $TOKENIZER -l $l -no-escape -a -threads $WORKERS | \
            $LC > $tok
    # perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    tok=$prep/test.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        $REPLACE_UNICODE_PUNCT | \
            $NORM_PUNC -l $l | \
            $REM_NON_PRINT_CHAR | \
            $TOKENIZER -l $l -no-escape -a -threads $WORKERS | \
            $LC > $tok
    # perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

for split in train valid; do
    if [ -f $tmp/$split.$src ] && [ -f $tmp/$split.$tgt ]; then
        echo "cleaned data found: $tmp/$split.$src & $tgt, skipping clean_corpus_n"    
    else
        perl $CLEAN -ratio 1.5 $prep/$split $src $tgt $tmp/$split 1 250
    fi
done

for L in $src $tgt; do
    cp $prep/test.$L $tmp/test.$L
done