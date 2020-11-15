#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL
raw=$RAW
prep=$CACHE/prep
ready=$CACHE/ready
lang=zh-en

mkdir -p $prep $ready

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

cd $raw

echo "pre-processing train data..."
for l in $src $tgt; do
    f=news-commentary-v12.$lang.$l
    if [ -f $prep/train.dirty.$l ]; then
        echo "$raw/training/$f found, skipping tokenizer"
    else
        if [[ $l == "zh" ]]; then            
            cat $raw/training/$f | \
                $REM_NON_PRINT_CHAR | \
                python -m jieba -d > $prep/train.dirty.$l
        else
            cat $raw/training/$f | \
                $REM_NON_PRINT_CHAR > $prep/train.dirty.$l
        fi
    fi
done
perl $CLEAN -ratio 9 $prep/train.dirty $src $tgt $prep/train 1 250 # 9 is default

echo "pre-processing valid data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $raw/dev/newsdev2017-$src$tgt-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $prep/valid.raw.$l

    if [[ $l == "zh" ]]; then
        cat $prep/valid.raw.$l | \
            $REM_NON_PRINT_CHAR | \
            python -m jieba -d > $prep/valid.$l
    else
        cat $prep/valid.raw.$l | \
            $REM_NON_PRINT_CHAR > $prep/valid.$l
    fi
    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $raw/newstest2017-$src$tgt-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $prep/test.raw.$l

    if [[ $l == "zh" ]]; then
        cat $prep/test.raw.$l | \
            $REM_NON_PRINT_CHAR | \
            python -m jieba -d > $prep/test.$l
    else
        cat $prep/test.raw.$l | \
            $REM_NON_PRINT_CHAR > $prep/test.$l
    fi
    echo ""
done