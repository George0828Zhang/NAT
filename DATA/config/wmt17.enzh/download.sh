#!/usr/bin/env bash
URLS=(
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://data.statmt.org/wmt17/translation-task/test-update-1.tgz"
)
FILES=(
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-update-1.tgz"
)

mkdir -p $RAW

echo "Downloading data ..."
cd $RAW
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
    tar zxvf $file
done