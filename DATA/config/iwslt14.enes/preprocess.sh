#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready
lang=en-es

mkdir -p $prep $ready

cd $raw

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $raw/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    grep -v '</title>' | \
    grep -v '</description>' > $raw/$f
    echo ""
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $raw/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$raw/${fname%.*}
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
    awk '{if (NR%23 == 0)  print $0; }' $raw/train.tags.$lang.$l > $prep/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $raw/train.tags.$lang.$l > $prep/train.$l

    cat $raw/IWSLT14.TED.dev2010.$lang.$l \
        $raw/IWSLT14.TED.tst2010.$lang.$l \
        $raw/IWSLT14.TED.tst2011.$lang.$l \
        $raw/IWSLT14.TED.tst2012.$lang.$l \
        > $prep/test.$l
done