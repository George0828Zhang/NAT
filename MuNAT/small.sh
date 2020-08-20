#!/usr/bin/env bash
TASK=mutual
DATA=../DATA/data-bin/iwslt14-enes

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
ulimit -n $(ulimit -Hn)
export CUDA_VISIBLE_DEVICES=0,1

fairseq-train --task translation_lev_bleu --user-dir .. \
    --noise random_mask \
    -s en -t es \
    --max-tokens 4096 \
    --max-tokens-valid 512 \
    --empty-cache-freq 1 \
    --update-freq 8 \
    --arch mutual_cmlm_small \
    --peer-type ar \
    --share-all-embeddings \
    --apply-bert-init \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --sg-length-pred \
    --criterion mutual_loss \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --lr 1.25e-4 \
    --clip-norm 0.0 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 3}' \
    --eval-bleu-detok space \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 8 \
    --max-update 50000 \
    --keep-last-epochs 3 \
    --log-format simple --log-interval 10 \
    --fp16 \
    $DATA
