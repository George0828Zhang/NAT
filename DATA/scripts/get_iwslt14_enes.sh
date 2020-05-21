#!/usr/bin/env bash
PREFIX=$1 #/media/george/Storage/DATA
TMP=$PREFIX/iwslt14.en-es.raw
mkdir -p $TMP

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
cd $TMP

URL="https://wit3.fbk.eu/archive/2014-01/texts/en/es/en-es.tgz"
GZ=en-es.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=es
lang=en-es
tmp=$TMP/tmp
mkdir -p $tmp

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

    cat $TMP/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    grep -v '</title>' | \
    grep -v '</description>' | \
    perl $TOKENIZER -threads 8 -l $l > $TMP/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $TMP/train.tags.$lang.tok $src $tgt $TMP/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $TMP/train.tags.$lang.clean.$l > $TMP/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $TMP/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$TMP/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -l $l | \
        perl $LC > $f
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