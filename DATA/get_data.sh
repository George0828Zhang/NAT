#!/usr/bin/env bash

# config
DATASET=iwslt14 # {'wmt14', 'wmt17', 'iwslt14'}
WORKERS=4
OUTDIR=$DATASET-used-dict # if not set, use default value of dataset's name

BPE_CODE=data-bin/news.2017.en-de/code
# 'None', don't apply bpe
# 'Current', learn on current dataset
# other, use other as code

BPE_TOKENS=40000 # only used when learning BPE

# dictionary for binirize the data
DICT=data-bin/news.2017.en-de/dict.txt # if DICT='None', learning dict on current dataset

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git
BPEROOT=subword-nmt/subword_nmt

if [ $DATASET = 'wmt17' ]; then
    
    OUTDIR=${OUTDIR:='wmt17_en_de'}

    DATADIR='wmt17_en_de'
    DATASCRIPT='scripts/get_wmt17.sh'
    src=de
    tgt=en

elif [ $DATASET = 'wmt14' ]; then
    
    OUTDIR=${OUTDIR:='wmt14_en_de'}

    DATADIR='wmt14_en_de'
    DATASCRIPT='scripts/get_wmt17.sh --icml17'
    src=de
    tgt=en

elif [ $DATASET = "iwslt14" ]; then
    
    OUTDIR=${OUTDIR:='iwslt14.tokenized.de-en'}

    DATADIR='iwslt14.tokenized.de-en'
    DATASCRIPT='scripts/get_iwslt14.sh'
    src=de
    tgt=en

else
    echo "DATASET: $DATASET is not supported"
    exit
fi

if [ -d data-bin/$OUTDIR ]; then
    echo "data-bin/$OUTDIR is already existed. Please change the OUTDIR or remove data-bin/$OUTDIR"
    exit
fi

# check if dataset have already processed
exist=1
for split in train valid test; do
    for l in $src $tgt;do
        file=$DATADIR/tmp/$split.$l
        if [ -f $file ]; then
            echo "$file is exist"
        else
            echo "$file is not exist"
            exist=0
        fi
    done
done

if [ $exist = 0 ]; then

    echo "download and tokenize $DATASET at $DATADIR"
    bash $DATASCRIPT

fi

# learn BPE if needed
if [ $BPE_CODE = 'None' ]; then
    echo "Don't apply BPE"
    for i in train valid test; do
        for l in $src $tgt; do
            cp $DATADIR/tmp/$i.$l $DATADIR/$i.$l
        done
    done
else
    if [ $BPE_CODE = 'Current' ]; then

        echo "Didn't provide BPE code, learn on current dataset"
        BPE_CODE=$DATADIR/tmp/code

        if [ -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE is exist, skip learning"
        else
            bash learn_bpe.sh $DATADIR/tmp $src $tgt $BPE_TOKENS
        fi

    else

        if [ ! -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE not found!"
            exit
        fi

    fi

    echo "use bpe code at $BPE_CODE"

    for L in $src $tgt; do
        for f in train.$L valid.$L test.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $DATADIR/tmp/$f > $DATADIR/$f
        done
    done

fi



if [ ! $DICT = 'None' ] && [ ! -f $DICT ]; then
    echo "dictionary: $DICT not found!"
    exit
else
    echo "use dictionary $DICT"
fi

# create data-bin
preprocess_args=" \
    --source-lang $src --target-lang $tgt \
    --joined-dictionary \
    --bpe subword_nmt \
    --trainpref $DATADIR/train --validpref $DATADIR/valid --testpref $DATADIR/test \
    --destdir data-bin/$OUTDIR \
    --workers $WORKERS \
"

if [ ! $DICT = 'None' ]; then
    preprocess_args="$preprocess_args --srcdict $DICT"
fi

fairseq-preprocess $preprocess_args