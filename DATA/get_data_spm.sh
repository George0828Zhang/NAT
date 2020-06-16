#!/usr/bin/env bash
PATH=$HOME/.local/bin:$PATH
# config
DATASET=$1  # {'wmt14', 'wmt17', 'newscrawl'}
WORKERS=4
OUTDIR=$DATASET # if not set, use default value of dataset's name
PREFIX=/media/george/Storage/DATA # put . to use pwd

SPM_MODEL=./data-bin/newscrawl/spm
# 'None', don't apply bpe
# 'Current', learn on current dataset
# other, use other as code

N_TOKENS=40000 # only used when learning BPE

# dictionary for binirize the data
DICT=./data-bin/newscrawl/dict.txt # if DICT='None', learning dict on current dataset

if [ $DATASET = 'wmt17' ]; then
    
    OUTDIR=${OUTDIR:='wmt17_en_de'}

    DATADIR='wmt17_en_de'
    DATASCRIPT='scripts/spm/get_wmt17.sh'
    src=de
    tgt=en
    langs='de en'

elif [ $DATASET = 'wmt14' ]; then
    
    OUTDIR=${OUTDIR:='wmt14_en_de'}

    DATADIR='wmt14_en_de'
    DATASCRIPT='scripts/spm/get_wmt17.sh'
    EXTRAOPTIONS='--icml17'
    src=de
    tgt=en
    langs='de en'

elif [ $DATASET = "newscrawl" ]; then
    
    OUTDIR=${OUTDIR:='newscrawl'}

    DATADIR='newscrawl.raw'
    DATASCRIPT='scripts/spm/get_newscrawl_mono.sh'
    langs='cs' #'en de fr es cs'

else
    echo "DATASET: $DATASET is not supported"
    exit
fi

OUTDIR=$(pwd)/data-bin/$OUTDIR
# if [ -d $OUTDIR ]; then
#     echo "$OUTDIR is already existed. Please change the OUTDIR or remove $OUTDIR"
#     exit
# fi

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

    echo "download and preprocess $DATASET at $DATADIR"
    bash $DATASCRIPT $PREFIX $EXTRAOPTIONS

fi

# learn BPE if needed
if [ $SPM_MODEL = 'None' ]; then
    echo "Don't apply SPM"
    for i in train valid test; do
        for l in $langs; do
            cp $DATADIR/tmp/$i.$l $DATADIR/$i.$l
        done
    done
else
    if [ $SPM_MODEL = 'Current' ]; then

        echo "Didn't provide SPM model, learn on current dataset"
        SPM_MODEL=$DATADIR/tmp/spm
        BPE_TRAIN=$DATADIR/tmp/all.bpe-train

        if [ -f $SPM_MODEL.model ]; then
            echo "SPM model: $SPM_MODEL.model exists, skip learning"
        else            
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

            echo "spm_train on ${BPE_TRAIN}..."
            spm_train --input=$BPE_TRAIN \
                    --model_prefix=$SPM_MODEL \
                    --vocab_size=$N_TOKENS \
                    --character_coverage=1.0 \
                    --model_type=unigram \
                    --normalization_rule_name=nmt_nfkc_cf
            #######################################################
        fi

        for obj in model vocab; do
            target=$OUTDIR/spm.$obj
            if [ -f $target ]; then
                echo "$target found, skipping copy."
            else
                echo "copying $SPM_MODEL.$obj to $target."
                mkdir -p $OUTDIR
                cp $SPM_MODEL.$obj $target
            fi
        done
    else

        if [ ! -f $SPM_MODEL.model ]; then
            echo "SPM model: $SPM_MODEL.model not found!"
            exit
        fi

    fi

    echo "use SPM model at $SPM_MODEL.model"

    for L in $langs; do
        for f in train.$L valid.$L test.$L; do
            if [ -f $DATADIR/$f ]; then
                echo "found $DATADIR/$f, skipping spm_encode"
            else
                echo "spm_encode to ${f}..."
                spm_encode --model=$SPM_MODEL.model \
                    --output_format=piece \
                    < $DATADIR/tmp/$f > $DATADIR/$f
            fi
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
    --bpe sentencepiece \
    --trainpref $DATADIR/train --validpref $DATADIR/valid --testpref $DATADIR/test \
    --destdir $OUTDIR \
    --workers $WORKERS \
"

if [ ! $DICT = 'None' ]; then
    preprocess_args="$preprocess_args --srcdict $DICT"
fi


if [ $DATASET = "newscrawl" ] || [ $DATASET = 'europarl' ]; then
    echo "making union of vocab..."
    DICT=$DATADIR/vocab.uniq
    cut -f1 $SPM_MODEL.vocab | tail -n +4 | sed "s/$/ 100/g" > $DICT
        
    echo "dict size :"
    wc -l < $DICT

    for l in $langs; do \
        fairseq-preprocess \
        --only-source \
        --source-lang $l \
        --srcdict $DICT \
        --joined-dictionary \
        --bpe sentencepiece \
        --trainpref $DATADIR/train \
        --validpref $DATADIR/valid \
        --destdir $OUTDIR \
        --workers $WORKERS
    done

    # format for multilingual denoising.
    cd $OUTDIR
    for l in $langs; do \
        mkdir -p $l
        for SPLIT in train valid; do \
            mv $SPLIT.$l-None.$l.bin $l/$SPLIT.bin 
            mv $SPLIT.$l-None.$l.idx $l/$SPLIT.idx
        done
        cp dict.$l.txt dict.txt
    done
else
    fairseq-preprocess $preprocess_args
fi