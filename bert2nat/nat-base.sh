#!/usr/bin/env bash
TASK=bert2nat
DATA=../DATA/data-bin/wmt14.en-de
TEACHER=/media/george/Data/xlmr.base/trimmed_nomask

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
ulimit -n $(ulimit -Hn)
export CUDA_VISIBLE_DEVICES=0

fairseq-train --user-dir .. \
    --task translation_lev_bleu --noise no_noise \
    -s en -t de \
    --max-tokens 1000 \
    --max-tokens-valid 1024 \
    --update-freq 4 \
    --arch bert2nat_iwslt16 \
    --teacher-dir $TEACHER \
    --hint-from-layer 8 \
    --src-embedding-copy \
    --apply-bert-init \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --hint-loss-factor 0.1 \
    --sg-length-pred \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --lr 5.432e-4 \
    --clip-norm 0.0 \
    --dropout 0.28 \
    --weight-decay 0.0001 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --keep-last-epochs 1 \
    --save-interval-updates 10 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 8 \
    --max-update 1000000 \
    --patience 160 \
    --log-format simple --log-interval 50 \
    --fp16 \
    $DATA

    # --patience: 160 runs * 600 steps/run = 100k steps