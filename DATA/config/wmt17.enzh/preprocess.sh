#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready
lang=zh-en

mkdir -p $prep $ready

cd $raw

echo "pre-processing train/valid data..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $raw/training/news-commentary-v12.$lang.$l > $prep/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $raw/training/news-commentary-v12.$lang.$l > $prep/train.$l
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $raw/newstest2017-enzh-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $prep/test.$l
    echo ""
done