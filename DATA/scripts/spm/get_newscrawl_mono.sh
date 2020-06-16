#!/usr/bin/env bash
PREFIX=${1:='.'}
TMP=$PREFIX/newscrawl.raw
WORKERS=4

mkdir -p $TMP
langs="cs fr es de en"
# cs hi ru
years="2007 2008 2009 2010 2011 2012 2013"

tmp=$TMP/tmp
prep=$TMP/prep
orig=$TMP/orig
mkdir -p $tmp $orig $prep

URLS=()
FILES=()

for y in $years; do \
    for l in $langs; do \
        URLS+=("http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.${y}.${l}.shuffled.gz")
        FILES+=("news.${y}.${l}.shuffled")
    done
done

user=newscrawl
pass=acrawl4me
cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    GZ=$file.gz
    if [ -f $GZ ]; then
        echo "$GZ already exists, skipping download"
    elif [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget -O $GZ --user $user --password $pass "$url"
        if [ -f $GZ ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi        
    fi
    
    if [ -f $file ]; then
        echo "$file already exists, skipping unzip"
    else
        gunzip $GZ
    fi
done

# concat data
for l in $langs; do \
    raw=$prep/all.$l.raw
    if [ -f $raw ]; then
        echo "$raw already exists, skipping concat"
    else
        echo "concat $l files into $raw"
        for yr in $years; do
            cat $orig/news.$yr.$l.shuffled >> $raw
        done
    fi
done

# train valid split
for l in $langs; do \
    raw=$prep/all.$l.raw
    echo "splitting $raw into train/valid ..."
    if [ -f $tmp/valid.$l ] && [ -f $tmp/train.$l ]; then
        echo "${tmp}/valid.${l} ${tmp}/train.${l} found, skipping split"
    else
        cat $raw | awk '{if (NR < 3000)  print $0; }' > $tmp/valid.$l
        cat $raw | awk '{if (NR >= 3000)  print $0; }' > $tmp/train.$l
    fi
done