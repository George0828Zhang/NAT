#!/usr/bin/env bash

# config
DATASET=$1  # {'wmt14', 'wmt17', 'iwslt14', 'iwslt14.en-es'}
WORKERS=4
OUTDIR=$DATASET # if not set, use default value of dataset's name
PREFIX=/media/george/Storage/DATA # put . to use pwd

BPE_CODE=Current
# 'None', don't apply bpe
# 'Current', learn on current dataset
# other, use other as code

BPE_TOKENS=80000 # only used when learning BPE

# dictionary for binirize the data
DICT=None # if DICT='None', learning dict on current dataset

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git
BPEROOT=subword-nmt/subword_nmt

if [ $DATASET = 'wmt17' ]; then
    
    OUTDIR=${OUTDIR:='wmt17_en_de'}

    DATADIR='wmt17_en_de'
    DATASCRIPT='scripts/get_wmt17.sh'
    src=de
    tgt=en
    langs='de en'

elif [ $DATASET = 'wmt14' ]; then
    
    OUTDIR=${OUTDIR:='wmt14_en_de'}

    DATADIR='wmt14_en_de'
    DATASCRIPT='scripts/get_wmt17.sh --icml17'
    src=de
    tgt=en
    langs='de en'

elif [ $DATASET = "iwslt14" ]; then
    
    OUTDIR=${OUTDIR:='iwslt14.tokenized.de-en'}

    DATADIR='iwslt14.tokenized.de-en'
    DATASCRIPT='scripts/get_iwslt14.sh'
    src=de
    tgt=en
    langs='de en'

elif [ $DATASET = "iwslt14.en-es" ]; then
    
    OUTDIR=${OUTDIR:='iwslt14.en-es'}

    DATADIR='iwslt14.en-es.raw'
    DATASCRIPT='scripts/get_iwslt14_enes.sh'
    src=es
    tgt=en
    langs='es en'

elif [ $DATASET = "newscrawl" ]; then
    
    OUTDIR=${OUTDIR:='newscrawl'}

    DATADIR='newscrawl.raw'
    DATASCRIPT='scripts/get_newscrawl_mono.sh'
    langs='en de fr es'

else
    echo "DATASET: $DATASET is not supported"
    exit
fi

if [ -d data-bin/$OUTDIR ]; then
    echo "data-bin/$OUTDIR is already existed. Please change the OUTDIR or remove data-bin/$OUTDIR"
    exit
fi

PREFIX=${PREFIX:='.'} # just in case
DATADIR=$PREFIX/$DATADIR

# check if dataset have already processed
exist=1
for split in train valid test; do
    for l in $langs;do
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
    bash $DATASCRIPT $PREFIX

fi

# learn BPE if needed
if [ $BPE_CODE = 'None' ]; then
    echo "Don't apply BPE"
    for i in train valid test; do
        for l in $langs; do
            cp $DATADIR/tmp/$i.$l $DATADIR/$i.$l
        done
    done
else
    if [ $BPE_CODE = 'Current' ]; then

        echo "Didn't provide BPE code, learn on current dataset"
        BPE_CODE=$DATADIR/tmp/code
        BPE_TRAIN=$DATADIR/tmp/all.bpe-train

        if [ -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE is exist, skip learning"
        else
            # bash learn_bpe.sh $DATADIR/tmp $src $tgt $BPE_TOKENS

            if [ -f $BPE_TRAIN ]; then
                echo "${BPE_TRAIN} found, skipping concat."
            else
                for l in $langs; do \
                    default=1000000
                    total=$(cat $DATADIR/tmp/train.${l} $DATADIR/tmp/valid.${l} | wc -l)
                    echo "lang $l total: $total."
                    if [ "$total" -gt "$default" ]; then
                        cat $DATADIR/tmp/train.${l} $DATADIR/tmp/valid.${l} | \
                        shuf -r -n $default >> $BPE_TRAIN
                    else
                        cat $DATADIR/tmp/train.${l} $DATADIR/tmp/valid.${l} >> $BPE_TRAIN
                    fi                    
                done
            fi

            echo "learn_bpe.py on ${BPE_TRAIN}..."
            python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $BPE_TRAIN > $BPE_CODE
            mkdir -p data-bin/$OUTDIR
            cp $BPE_CODE data-bin/$OUTDIR/code
            #######################################################
        fi

    else

        if [ ! -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE not found!"
            exit
        fi

    fi

    echo "use bpe code at $BPE_CODE"

    for L in $langs; do
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


if [ $DATASET = "newscrawl" ]; then
    echo "making union of vocab..."
    mkdir -p data-bin/$OUTDIR
    DICT=$DATADIR/vocab.all
    rm -f $DICT
    # create own dict for each lang first. 
    for l in $langs; do \
        echo "get $l vocab..."
        cat $DATADIR/train.$l $DATADIR/valid.$l | python $BPEROOT/get_vocab.py | cut -d ' ' -f1 >> $DICT
    done

    # forge fake vocab from all langs
    sort -u $DICT | sed "s/$/ 100/g" > data-bin/$OUTDIR/dict.txt
    DICT=data-bin/$OUTDIR/dict.txt
    echo "dict size :"
    wc -l $DICT

    for l in $langs; do \
        fairseq-preprocess \
        --source-lang $l \
        --srcdict $DICT \
        --task cross_lingual_lm \
        --only-source \
        --trainpref $DATADIR/train \
        --validpref $DATADIR/valid \
        --destdir data-bin/$OUTDIR \
        --workers $WORKERS
    done

    # format for multilingual denoising.
    cd data-bin/$OUTDIR
    for l in $langs; do \
        mkdir -p $l
        for SPLIT in train valid; do \
            mv $SPLIT.$l-None.$l.bin $l/$SPLIT.bin 
            mv $SPLIT.$l-None.$l.idx $l/$SPLIT.idx
        done
        # cp dict.$l.txt dict.txt
    done
else
    fairseq-preprocess $preprocess_args
fi