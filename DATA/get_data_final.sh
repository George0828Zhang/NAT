#!/usr/bin/env bash
# config
export PREFIX=/media/george/Data
export DATABIN=$(pwd)/data-bin
export CONFIG=config/multi30k
export SRCLANG=de
export TGTLANG=en
export WORKERS=4
export BPE_TOKENS=4000

# setup path
source $CONFIG/path.sh

# check
if [[ -d $OUTDIR ]]; then
    echo "$OUTDIR already exists. Please change the OUTDIR or remove $OUTDIR"
    exit 1
fi

# download data
bash $CONFIG/download.sh

# preprocess data
bash $CONFIG/preprocess.sh

# (train, )apply spm/bpe
bash learn_or_apply_bpe.sh

# binarize
mkdir -p $OUTDIR
LOCS=""
for split in train test valid; do
    if [[ -f $CACHE/ready/${split}.$SRCLANG ]]; then
        LOCS="$LOCS --${split}pref $CACHE/ready/${split}"
    fi
done
fairseq-preprocess \
    --source-lang $SRCLANG \
    --target-lang $TGTLANG \
    $LOCS \
    --destdir $OUTDIR \
    --workers $WORKERS \
    $BINARIZEARGS