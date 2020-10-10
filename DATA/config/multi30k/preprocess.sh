#!/usr/bin/env bash
src=$SRCLANG
tgt=$TGTLANG
distill=$DISTILL

raw=$RAW/iwslt/en-de
prep=$CACHE/prep
ready=$CACHE/ready

mkdir -p $prep $ready

DIR=$RAW/dataset/data/task1/tok/

echo "creating train, valid, test..."
for l in $src $tgt; do    
    cp $DIR/train.lc.norm.tok.$l $prep/train.$l
    cp $DIR/val.lc.norm.tok.$l $prep/valid.$l
    cp $DIR/test_2016_flickr.lc.norm.tok.$l $prep/test.$l
done
