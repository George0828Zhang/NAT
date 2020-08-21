#!/usr/bin/env bash
PREFIX=${1:='.'}
WORKERS=4
TMP=$PREFIX/iwslt14.en-es.raw
mkdir -p $TMP

URL="https://wit3.fbk.eu/archive/2014-01/texts/en/es/en-es.tgz"
GZ=en-es.tgz

src=en
tgt=es
lang=en-es
tmp=$TMP/tmp
# prep=$TMP/prep
orig=$TMP/orig
mkdir -p $tmp $orig $prep

cd $orig

echo "Downloading data from ${URL}..."
if [ -f $GZ ]; then
    echo "Data already downloaded."
else
    wget "$URL"
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    grep -v '</title>' | \
    grep -v '</description>' > $TMP/$f
    # perl $TOKENIZER -threads 8 -l $l > $TMP/$tok
    echo ""
done


echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$TMP/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" > $f
        # perl $TOKENIZER -threads 8 -l $l | \
        # perl $LC > $f
        echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $TMP/train.tags.$lang.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $TMP/train.tags.$lang.$l > $tmp/train.$l

    cat $TMP/IWSLT14.TED.dev2010.$lang.$l \
        $TMP/IWSLT14.TED.tst2010.$lang.$l \
        $TMP/IWSLT14.TED.tst2011.$lang.$l \
        $TMP/IWSLT14.TED.tst2012.$lang.$l \
        > $tmp/test.$l
done