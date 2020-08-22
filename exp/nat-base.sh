#!/usr/bin/env bash
TASK=nat_base
DATA=../DATA/data-bin/iwslt16.distill.en-de

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
ulimit -n $(ulimit -Hn)
export CUDA_VISIBLE_DEVICES=1

fairseq-train --task translation_lev_bleu --user-dir .. \
    --noise full_mask \
    -s en -t de \
    --max-tokens 2048 \
    --max-tokens-valid 1024 \
    --update-freq 1 \
    --arch na_transformer_iwslt16 \
    --apply-bert-init \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --sg-length-pred \
    --criterion nat_loss \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 746 \
    --lr 2e-4 \
    --clip-norm 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0}' \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 8 \
    --max-update 300000 \
    --log-format simple --log-interval 50 \
    $DATA

    # --no-epoch-checkpoints \
    # --save-interval-updates 150 --keep-interval-updates 1 \