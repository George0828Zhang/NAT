#!/usr/bin/env bash
bash $FAIRSEQDIR/examples/language_model/prepare-wikitext-103.sh

TEXT=wikitext-103
VOCAB=data-bin/iwslt14.tokenized.de-en/dict.en.txt
fairseq-preprocess \
    --srcdict $VOCAB\
    --joined-dictionary \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 4