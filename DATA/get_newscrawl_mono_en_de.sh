#!/usr/bin/env bash
FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
ENCODER=$FAIRSEQDIR/examples/roberta/multiprocessing_bpe_encoder.py
WORKERS=4

bash get_gpt2_bpe.sh


URLS=(
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)
FILES=(
    "news.2017.de.shuffled.deduped.gz"
    "news.2017.en.shuffled.deduped.gz"
)

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
        elif [ ${file: -3} == ".gz" ]; then
            gunzip $file
        fi
    fi
done


TMP=news.2017.tmp
mkdir -p $TMP
for l in en de; do \
    TRAIN=$TMP/news.2017.train.${l}
    VALID=$TMP/news.2017.valid.${l}
    if [ -f $VALID ]; then
        echo "${VALID} found, skipping split."
    else
        awk '{if (NR%3291 == 51)  print $0; }' news.2017.${l}.shuffled.deduped > $VALID
    fi
    if [ -f $TRAIN ]; then
        echo "${TRAIN} found, skipping split."
    else
        awk '{if (NR%3291 != 51)  print $0; }' news.2017.${l}.shuffled.deduped > $TRAIN
    fi
done

for SPLIT in train valid; do \
    for l in en de; do \
        if [ -f $TMP/news.2017.${SPLIT}.${l}.bpe ]; then
            echo "${SPLIT}.${l}.bpe found, skipping encode."
        else
            python $ENCODER \
            --encoder-json gpt2_bpe/encoder.json \
            --vocab-bpe gpt2_bpe/vocab.bpe \
            --inputs $TMP/news.2017.${SPLIT}.${l} \
            --outputs $TMP/news.2017.${SPLIT}.${l}.bpe \
            --keep-empty \
            --workers $WORKERS
        fi
    done
done

for l in en de; do \
    fairseq-preprocess \
        --only-source \
        --srcdict gpt2_bpe/dict.txt \
        --trainpref $TMP/news.2017.train.${l}.bpe \
        --validpref $TMP/news.2017.valid.${l}.bpe \
        --destdir data-bin/news.2017.en-de/${l} \
        --workers $WORKERS
done
cp gpt2_bpe/dict.txt data-bin/news.2017.en-de/dict.txt