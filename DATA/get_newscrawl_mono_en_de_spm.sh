#!/usr/bin/env bash
FAIRSEQDIR:=$HOME/Projects/NATLab/fairseq
ENCODER=$FAIRSEQDIR/examples/roberta/multiprocessing_bpe_encoder.py
WORKERS=4

BPE_TOKENS=50000


URLS=(
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)
FILES=(
    "news.2017.raw/news.2017.de.shuffled.deduped.gz"
    "news.2017.raw/news.2017.en.shuffled.deduped.gz"
)
mkdir -p news.2017.raw
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
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip $file
        fi
    fi
done


TMP=news.2017.tmp
mkdir -p $TMP
for l in en de; do \
    SOURCE=news.2017.raw/news.2017.${l}.shuffled.deduped
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
        awk '{if (NR%3291 < 50)  print $0; }' $SOURCE > $TRAIN
        #awk '{if (NR%3291 != 50)  print $0; }' $SOURCE > $TRAIN
    fi
done

TRAIN=$TMP/news.2017.bpe.train
SPM_MODEL=$TMP/news.2017

if [ -f $TRAIN ]; then
    echo "${TRAIN} found, skipping concat."
else
    for l in en de; do \
        # cat news.2017.raw/news.2017.${l}.shuffled.deduped >> $TRAIN
        shuf -r -n 10000000 news.2017.raw/news.2017.${l}.shuffled.deduped >> $TRAIN
    done
fi

echo "spm_train on ${TRAIN}..."
if [ -f $SPM_MODEL.model ]; then
    echo "${SPM_MODEL}.model found, skipping learn."
else
    spm_train --input=$TRAIN \
    --model_prefix=$SPM_MODEL \
    --vocab_size=$BPE_TOKENS \
    --character_coverage=1.0 \
    --model_type=bpe
    #--input_sentence_size=1000000
fi
for SPLIT in train valid; do \
    for l in en de; do \
        src=$TMP/news.2017.${SPLIT}.${l}
        bpe=$TMP/news.2017.${SPLIT}.${l}.bpe
        echo "spm_encode to ${src}..."
        if [ -f $bpe ]; then
            echo "$bpe found, skipping apply."
        else
            spm_encode --model=$SPM_MODEL.model \
            --output_format=piece \
            < $src > $bpe            
        fi
    done
done

mkdir -p $TMP/news.2017.bpe
for SPLIT in train valid; do \
    for l in en de; do \
        echo "moved to ${TMP}/news.2017.bpe/${SPLIT}.${l}..."
        mv $TMP/news.2017.${SPLIT}.${l}.bpe $TMP/news.2017.bpe/${SPLIT}.${l}
    done
done

TEXT=$TMP/news.2017.bpe
fairseq-preprocess --source-lang de --target-lang en \
    --joined-dictionary \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --destdir data-bin/news.2017.en-de.spm \
    --workers $WORKERS

cp -r $SPM_MODEL.model data-bin/news.2017.en-de.spm/news.2017.model
cp -r $SPM_MODEL.vocab data-bin/news.2017.en-de.spm/news.2017.vocab
cd data-bin/news.2017.en-de.spm
for l in en de; do \
    mkdir -p ${l}
    for SPLIT in train valid; do \
        mv ${SPLIT}.de-en.${l}.bin ${l}/${SPLIT}.bin 
        mv ${SPLIT}.de-en.${l}.idx ${l}/${SPLIT}.idx
    done
    cp dict.${l}.txt dict.txt
done