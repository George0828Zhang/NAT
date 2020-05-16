#!/usr/bin/env bash
WORKERS=4
BPE_TOKENS=32000
TMP=/media/george/Storage/DATA/news.2017.tmp
RAW=/media/george/Storage/DATA/news.2017.raw
BIN=data-bin
OUTDIR=$BIN/news.2017.spm
## there will be SPM_MODEL.model and SPM_MODEL.vocab 
SPM_MODEL=$BIN/news.2017.spm

mkdir -p $TMP
mkdir -p $BIN
mkdir -p $RAW

URLS=(
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)
FILES=(
    "$RAW/news.2017.de.shuffled.deduped.gz"
    "$RAW/news.2017.en.shuffled.deduped.gz"
)
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget -O $file "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi        
    fi

    if [ ${file: -4} == ".tgz" ]; then
        tar zxvf $file
    elif [ ${file: -4} == ".tar" ]; then
        tar xvf $file
    elif [ ${file: -4} == ".zip" ]; then
        unzip $file
    elif [ ${file: -3} == ".gz" ]; then
        gunzip $file
    fi
done

## make training data for spm_train
TRAIN=$TMP/news.2017.bpe.train
if [ -f $TRAIN ]; then
    echo "${TRAIN} found, skipping concat."
else
    echo "make training data for spm_train"
    for l in en de; do \
        ## sample this many sentence from each lang to train spm
        shuf -r -n 10000000 $RAW/news.2017.${l}.shuffled.deduped >> $TRAIN
    done
    echo "shuffling ${TRAIN}..."
    cat $TRAIN | shuf -o $TRAIN
fi


## spm_train
echo "spm_train on ${TRAIN}..."
if [ -f $SPM_MODEL.model ]; then
    echo "${SPM_MODEL}.model found, skipping learn."
else
    spm_train --input=$TRAIN \
    --model_prefix=$SPM_MODEL \
    --vocab_size=$BPE_TOKENS \
    --character_coverage=1.0 \
    --normalization_rule_name=nmt_nfkc_cf \
    --model_type=bpe
    #--input_sentence_size=1000000
fi

## train-valid split
for l in en de; do \
    SOURCE=$RAW/news.2017.${l}.shuffled.deduped
    TRAIN=$TMP/news.2017.train.${l}
    VALID=$TMP/news.2017.valid.${l}
    if [ -f $VALID ]; then
        echo "${VALID} found, skipping split."
    else
        awk '{if (NR%3291 == 51)  print $0; }' $SOURCE > $VALID
    fi
    if [ -f $TRAIN ]; then
        echo "${TRAIN} found, skipping split."
    else
        awk '{if (NR%3291 != 51)  print $0; }' $SOURCE > $TRAIN
    fi
done

# get language specific vocabulary
# for l in en de; do \
#     echo "generating ${SPM_MODEL}.vocab.${l}"
#     SOURCE=$RAW/news.2017.${l}.shuffled.deduped
#     VOCAB=$SPM_MODEL.vocab.${l}
    
#     if [ -f $VOCAB ]; then
#         echo "${VOCAB} found, skipping generate vocab."
#     else
#         spm_encode --model=$SPM_MODEL.model --generate_vocabulary \
#         < $SOURCE > $VOCAB
#     fi
# done

## spm_encode
mkdir -p $TMP/news.2017.bpe
for SPLIT in train valid; do \
    for l in en de; do \
        src=$TMP/news.2017.${SPLIT}.${l}
        bpe=$TMP/news.2017.bpe/${SPLIT}.${l}
        # vocab=$SPM_MODEL.vocab.${l}
        echo "spm_encode to ${src}..."
        if [ -f $bpe ]; then
            echo "$bpe found, skipping apply."
        else
            # spm_encode --model=$SPM_MODEL.model \
            # --output_format=piece \
            # --vocabulary=$vocab --vocabulary_threshold=50 \
            # < $src > $bpe    
            spm_encode --model=$SPM_MODEL.model \
            --output_format=piece \
            < $src > $bpe    
        fi
    done
done


## binarize
if [ -d $OUTDIR ]; then
    echo "$OUTDIR found, skipping preprocess."
else
    mkdir -p $OUTDIR
    
    ## make spm vocab into fairseq vocab
    cut -f1 $SPM_MODEL.vocab | tail -n +4 | sed "s/$/ 100/g" > $OUTDIR/dict.txt

    TEXT=$TMP/news.2017.bpe
    fairseq-preprocess --source-lang de --target-lang en \
        --joined-dictionary \
        --srcdict $OUTDIR/dict.txt \
        --trainpref $TEXT/train \
        --validpref $TEXT/valid \
        --destdir $OUTDIR \
        --workers $WORKERS
fi

## format for multilingual denoising.
cd $OUTDIR
for l in en de; do \
    mkdir -p ${l}
    for SPLIT in train valid; do \
        mv ${SPLIT}.de-en.${l}.bin ${l}/${SPLIT}.bin 
        mv ${SPLIT}.de-en.${l}.idx ${l}/${SPLIT}.idx
    done
    # cp dict.${l}.txt dict.txt
done