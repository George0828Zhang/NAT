#!/usr/bin/env bash
PREFIX=${1:='.'}
TMP=$PREFIX/newscrawl.raw
WORKERS=4

mkdir -p $TMP
langs="fr es de en"
# cs hi ru
years="2007 2008"

cd $PREFIX
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=$(pwd)/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
REPLACE_UNICODE_PUNCT=$SCRIPTS/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
# CLEAN=$SCRIPTS/training/clean-corpus-n.perl

# echo 'Cloning wmt16-scripts github repository (for accent scripts)...'
# git clone https://github.com/rsennrich/wmt16-scripts.git
# WMT16_SCRIPTS=$(pwd)/wmt16-scripts
# NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
# REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# echo 'Cloning XLM github repository (for lowercase remove accent scripts)...'
# git clone https://github.com/facebookresearch/XLM.git
# LOWER_REMOVE_ACCENT=$(pwd)/XLM/tools/lowercase_and_remove_accent.py

cd $TMP

# rm -f vocab_xnli_15 codes_xnli_15
# wget https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15
# wget https://dl.fbaipublicfiles.com/XLM/codes_xnli_15
# cat codes_xnli_15 | cut -d ' ' -f1,2 > codes_xnli_15

URLS=()
FILES=()
    
for y in $years; do \
    for l in $langs; do \
        URLS+=("http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.${y}.${l}.shuffled.gz")
        FILES+=("$TMP/news.${y}.${l}.shuffled")
    done
done

user=newscrawl
pass=acrawl4me
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

# tokenize and clean data
tmp=$TMP/tmp
mkdir -p $tmp
for l in $langs; do \
    raw=$tmp/all.$l.raw
    tok=$tmp/all.$l.tok

    echo "concat $l files into $raw"
    rm -f $raw
    for yr in $years; do
        cat $TMP/news.$yr.$l.shuffled >> $raw
    done

    if [ -f $tok ]; then
        echo "$tok already exists, skipping concat"
    else
        echo "preprocessing $raw ..."
        cat $raw | \
            $REPLACE_UNICODE_PUNCT | \
            $NORM_PUNC -l $l | \
            $REM_NON_PRINT_CHAR | \
            $TOKENIZER -l $l -no-escape -threads $WORKERS | \
            $LC > $tok
            # python $LOWER_REMOVE_ACCENT > $tok
            # $REMOVE_DIACRITICS | \ maybe not suitable for generation task.
    fi    
done

# train valid split --additional-suffix
for l in $langs; do \
    tok=$tmp/all.$l.tok
    echo "splitting $tok into train/valid ..."
    if [ -f $tmp/valid.$l ] && [ -f $tmp/train.$l ]; then
        echo "${tmp}/valid.${l} ${tmp}/train.${l} found, skipping split"
    else
        cat $tok | awk '{if (NR < 3000)  print $0; }' > $tmp/valid.$l
        cat $tok | awk '{if (NR >= 3000)  print $0; }' > $tmp/train.$l
    fi
done

# # binarize
# if [ -d $OUTDIR ]; then
#     echo "$OUTDIR found, skipping preprocess."
# else
#     mkdir -p $OUTDIR
#     ## make spm vocab into fairseq vocab
#     cut -f1 $SPM_MODEL.vocab | tail -n +4 | sed "s/$/ 100/g" > $OUTDIR/dict.txt

#     TEXT=$TMP/news.2011.bpe
#     fairseq-preprocess --source-lang de --target-lang en \
#         --joined-dictionary \
#         --srcdict $OUTDIR/dict.txt \
#         --trainpref $TEXT/train \
#         --validpref $TEXT/valid \
#         --destdir $OUTDIR \
#         --workers $WORKERS
# fi
# # format for multilingual denoising.
# cd $OUTDIR
# for l in en de; do \
#     mkdir -p ${l}
#     for SPLIT in train valid; do \
#         mv ${SPLIT}.de-en.${l}.bin ${l}/${SPLIT}.bin 
#         mv ${SPLIT}.de-en.${l}.idx ${l}/${SPLIT}.idx
#     done
#     # cp dict.${l}.txt dict.txt
# done