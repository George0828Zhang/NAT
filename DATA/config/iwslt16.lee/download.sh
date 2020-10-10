#!/usr/bin/env bash
URL="https://drive.google.com/u/0/uc?id=1m7dZqEXHWPYcre6xxsFwFLrb9CRCZGmn&export=download"
GZ=iwslt.tar.gz

mkdir -p $RAW

echo "Downloading data from ${URL}..."
cd $RAW
if [ -f $GZ ]; then
    echo "Data already downloaded."
else
    gdown "$URL"
fi

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

if [[ -d $RAW/iwslt/en-de ]]; then
    echo "Data already unzipped."
else
    tar zxvf $GZ
fi
# tree $RAW/iwslt/en-de