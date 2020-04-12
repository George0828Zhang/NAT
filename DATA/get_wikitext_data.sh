#!/usr/bin/env bash
#FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
ENCODER=$FAIRSEQDIR/examples/roberta/multiprocessing_bpe_encoder.py
WORKERS=4

bash get_gpt2_bpe.sh

URLS=(
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
)
FILES=(
    "wikitext-103-raw-v1.zip"
)

mkdir -p gpt2_bpe
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

for SPLIT in train valid test; do \
    python $ENCODER \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers $WORKERS; \
done

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers $WORKERS