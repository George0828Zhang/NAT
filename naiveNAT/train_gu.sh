#!/usr/bin/env bash
TASK=gu
DATA=/hdd/NAT/data-bin/iwslt14.tokenized.de-en
mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train --task translation_lev_bleu --noise no_noise --user-dir . \
    -s de -t en \
    --max-tokens 4000 \
    --max-tokens-valid 1000 \
    --empty-cache-freq 1 \
    --update-freq 8 \
    --arch gu \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --apply-bert-init \
    --src-embedding-copy \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --sg-length-pred \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --lr 5e-4 \
    --clip-norm 0.0 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 4 \
    --memory-efficient-fp16 \
    --max-update 20000 \
    --log-format simple --log-interval 1 \
    $DATA

    # --share-all-embeddings \
