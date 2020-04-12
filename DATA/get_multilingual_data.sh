#!/usr/bin/env bash
FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
ENCODER=$FAIRSEQDIR/examples/roberta/multiprocessing_bpe_encoder.py
WORKERS=4
XLM_DIR=$PWD/XLM
WIKI_PATH=$XLM_DIR/data/wiki

echo 'Getting gpt2 bpe cpde...'
bash get_gpt2_bpe.sh


# Download XLM
if [ ! -d "$XLM_DIR" ]; then
  echo "Cloning XLM from GitHub repository..."
  git clone https://github.com/facebookresearch/XLM
fi
cd XLM

for lg in de en es fr; do
    echo 'Getting data from lang ${lg}'
    bash get-data-wiki.sh $lg
done


for lg in de en es fr; do
    for SPLIT in train valid test; do \
        python $ENCODER \
            --encoder-json gpt2_bpe/encoder.json \
            --vocab-bpe gpt2_bpe/vocab.bpe \
            --inputs $WIKI_PATH/txt/${lg}.${SPLIT} \
            --outputs $WIKI_PATH/txt/${lg}.${SPLIT}.bpe \
            --keep-empty \
            --workers $WORKERS; \
    done

    fairseq-preprocess \
        --only-source \
        --srcdict gpt2_bpe/dict.txt \
        --trainpref $WIKI_PATH/txt/${lg}.train.bpe \
        --validpref $WIKI_PATH/txt/${lg}.valid.bpe \
        --testpref $WIKI_PATH/txt/${lg}.test.bpe \
        --destdir data-bin/wikitext-103 \
        --workers $WORKERS
done

