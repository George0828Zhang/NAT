#!/usr/bin/env bash
URLS=(
    "http://dl.fbaipublicfiles.com/nat/distill_dataset.zip"
    "http://dl.fbaipublicfiles.com/nat/original_dataset.zip"
)
FILES=(
    "distill_dataset.zip"
    "original_dataset.zip"
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
        unzip $file
    fi
done