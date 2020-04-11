#!/usr/bin/env bash
bash $FAIRSEQDIR/examples/translation/prepare-iwslt14.sh

TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 4