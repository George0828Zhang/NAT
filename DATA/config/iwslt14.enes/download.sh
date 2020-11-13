#!/usr/bin/env bash
URLS=(
    "https://wit3.fbk.eu/archive/2014-01/texts/en/es/en-es.tgz"
)
FILES=(
    "en-es.tgz"
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