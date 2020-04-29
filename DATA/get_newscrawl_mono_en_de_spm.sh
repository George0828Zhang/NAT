#!/usr/bin/env bash
WORKERS=4
BPE_TOKENS=50000
TMP=news.2017.tmp
OUTDIR=data-bin/news.2017.en-de.spm
# there will be SPM_MODEL.model and SPM_MODEL.vocab 
SPM_MODEL=data-bin/news.2017.spm

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

# train-valid split
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
        awk '{if (NR%3291 != 51)  print $0; }' $SOURCE > $TRAIN
    fi
done

# make training data for spm_train
TRAIN=$TMP/news.2017.bpe.train
if [ -f $TRAIN ]; then
    echo "${TRAIN} found, skipping concat."
else
    for l in en de; do \
        # sample this many sentence from each lang to train spm
        shuf -r -n 10000000 news.2017.raw/news.2017.${l}.shuffled.deduped >> $TRAIN
    done
fi

# spm_train
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

# spm_encode
mkdir -p $TMP/news.2017.bpe
for SPLIT in train valid; do \
    for l in en de; do \
        src=$TMP/news.2017.${SPLIT}.${l}
        bpe=$TMP/news.2017.bpe/${SPLIT}.${l}
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

# binarize
if [ -d $OUTDIR ]; then
    echo "$OUTDIR found, skipping preprocess."
else
    TEXT=$TMP/news.2017.bpe
    fairseq-preprocess --source-lang de --target-lang en \
        --joined-dictionary \
        --trainpref $TEXT/train \
        --validpref $TEXT/valid \
        --destdir $OUTDIR \
        --workers $WORKERS
fi
# format for multilingual denoising.
cd $OUTDIR
for l in en de; do \
    mkdir -p ${l}
    for SPLIT in train valid; do \
        mv ${SPLIT}.de-en.${l}.bin ${l}/${SPLIT}.bin 
        mv ${SPLIT}.de-en.${l}.idx ${l}/${SPLIT}.idx
    done
    cp dict.${l}.txt dict.txt
done