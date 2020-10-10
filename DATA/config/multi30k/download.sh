#!/usr/bin/env bash
URL="https://github.com/multi30k/dataset"

mkdir -p $RAW
cd $RAW
echo "Downloading data from ${URL}..."
git clone ${URL}
