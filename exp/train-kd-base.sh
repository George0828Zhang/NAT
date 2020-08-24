#!/usr/bin/env bash
TASK=kd_base
DATA=../DATA/data-bin/iwslt16.distill.en-de
CHECK=checkpoints/ar_base/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
ulimit -n $(ulimit -Hn)
export CUDA_VISIBLE_DEVICES=0

fairseq-train --user-dir .. \
    --task translation_mutual \
    --noise full_mask \
    --freeze-peer \
    -s en -t de \
    --max-tokens 4096 \
    --max-tokens-valid 1024 \
    --update-freq 1 \
    --arch mutual_learn_nat_iwslt16 \
    --peer-type ar \
    --restore-file $CHECK \
    --reset-optimizer --reset-dataloader --reset-meters \
    --load-peer-only \
    --src-embedding-copy \
    --apply-bert-init \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --sg-length-pred \
    --criterion knowledge_distillation_loss \
    --kd-loss-factor 0.5 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 746 \
    --lr 5e-4 \
    --clip-norm 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 8 \
    --max-update 300000 \
    --log-format simple --log-interval 50 \
    --fp16 \
    $DATA