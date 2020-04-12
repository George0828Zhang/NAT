#!/usr/bin/env bash

URLS=(
    "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
    "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
    "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt"
)
FILES=(
    "gpt2_bpe/encoder.json"
    "gpt2_bpe/vocab.bpe"
    "gpt2_bpe/dict.txt"
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
    fi
done