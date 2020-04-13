#!/usr/bin/env bash
FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
ENCODER=$FAIRSEQDIR/examples/roberta/multiprocessing_bpe_encoder.py
WORKERS=4

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=50000


URLS=(
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)
FILES=(
    "news.2017.raw/news.2017.de.shuffled.deduped.gz"
    "news.2017.raw/news.2017.en.shuffled.deduped.gz"
)
mkdir -p news.2017.raw
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget -O $file "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip $file
        fi
    fi
done


TMP=news.2017.tmp
mkdir -p $TMP
for l in en de; do \
    SOURCE=news.2017.raw/news.2017.${l}.shuffled.deduped
    TRAIN=$TMP/news.2017.train.${l}
    VALID=$TMP/news.2017.valid.${l}
    if [ -f $VALID ]; then
        echo "${VALID} found, skipping split."
    else
        awk '{if (NR%3291 == 51)  print $0; }' $SOURCE > $VALID
    fi
    if [ -f $TRAIN ]; then
        echo "${TRAIN} found, skipping split."
    else
        awk '{if (NR%3291 != 50)  print $0; }' $SOURCE > $TRAIN
    fi
done


echo "pre-processing train data..."
for SPLIT in train valid; do \
    for l in en de; do \
        f=$TMP/news.2017.${SPLIT}.${l}
        tok=$TMP/news.2017.${SPLIT}.${l}.tok

        if [ -f $tok ]; then
            echo "${SPLIT}.${l}.tok found, skipping tokenization."
        else
            cat $f | \
            perl $TOKENIZER -threads $WORKERS -l $l | \
            perl $LC > $tok
        fi
    done
done

TRAIN=$TMP/news.2017.all.tok
BPE_CODE=$TMP/code
# rm -f $TRAIN

if [ -f $TRAIN ]; then
    echo "${TRAIN} found, skipping concat."
else
    for SPLIT in train valid; do \
        for l in en de; do \
            cat $TMP/news.2017.${SPLIT}.${l}.tok >> $TRAIN
        done
    done
fi

echo "learn_bpe.py on ${TRAIN}..."
if [ -f $BPE_CODE ]; then
    echo "${BPE_CODE} found, skipping learn."
else
    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
fi

for SPLIT in train valid; do \
    for l in en de; do \
        tok=$TMP/news.2017.${SPLIT}.${l}.tok
        bpe=$TMP/news.2017.${SPLIT}.${l}.tok.bpe
        echo "apply_bpe.py to ${tok}..."
        if [ -f $bpe ]; then
            echo "$bpe found, skipping apply."
        else
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tok > $bpe
        fi
    done
done

# for l in en de; do \
#     fairseq-preprocess \
#         --only-source \
#         --bpe subword_nmt \
#         --trainpref $TMP/news.2017.train.${l}.tok.bpe \
#         --validpref $TMP/news.2017.valid.${l}.tok.bpe \
#         --destdir data-bin/news.2017.en-de/${l} \
#         --workers $WORKERS
# done
# cd data-bin/news.2017.en-de
# cp de/dict.txt dict.txt

mkdir -p $TMP/news.2017.tok.bpe
for SPLIT in train valid; do \
    for l in en de; do \
        echo "moved to ${TMP}/news.2017.tok.bpe/${SPLIT}.${l}..."
        mv $TMP/news.2017.${SPLIT}.${l}.tok.bpe $TMP/news.2017.tok.bpe/${SPLIT}.${l}
    done
done

TEXT=$TMP/news.2017.tok.bpe
fairseq-preprocess --source-lang de --target-lang en \
    --joined-dictionary \
    --bpe subword_nmt \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --destdir data-bin/news.2017.en-de \
    --workers $WORKERS

cp $BPE_CODE data-bin/news.2017.en-de/code
cd data-bin/news.2017.en-de
for l in en de; do \
    mkdir -p ${l}
    for SPLIT in train valid; do \
        mv ${SPLIT}.de-en.${l}.bin ${l}/${SPLIT}.bin 
        mv ${SPLIT}.de-en.${l}.idx ${l}/${SPLIT}.idx
    done
    cp dict.${l}.txt dict.txt
done
# cp $BPE_CODE code


# for SPLIT in train valid; do \
#     for l in en de; do \
#         if [ -f $TMP/news.2017.${SPLIT}.${l}.bpe ]; then
#             echo "${SPLIT}.${l}.bpe found, skipping encode."
#         else
#             python $ENCODER \
#             --encoder-json gpt2_bpe/encoder.json \
#             --vocab-bpe gpt2_bpe/vocab.bpe \
#             --inputs $TMP/news.2017.${SPLIT}.${l} \
#             --outputs $TMP/news.2017.${SPLIT}.${l}.bpe \
#             --keep-empty \
#             --workers $WORKERS
#         fi
#     done
# done

# for l in en de; do \
#     fairseq-preprocess \
#         --only-source \
#         --srcdict gpt2_bpe/dict.txt \
#         --trainpref $TMP/news.2017.train.${l}.bpe \
#         --validpref $TMP/news.2017.valid.${l}.bpe \
#         --destdir data-bin/news.2017.en-de/${l} \
#         --workers $WORKERS
# done
# cp gpt2_bpe/dict.txt data-bin/news.2017.en-de/dict.txt