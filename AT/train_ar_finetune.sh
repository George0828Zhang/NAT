#!/usr/bin/env bash
TASK=ar_finetune
DATA=../DATA/data-bin/iwslt14.tokenized.de-en
CHECK=checkpoints/ar_denoise/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train --task translation_from_pretrained_bart --langs en,de\
    -s de -t en \
    --restore-file $CHECK \
    --reset-optimizer --reset-dataloader --reset-meters \
    --find-unused-parameters \
    --max-tokens 4096 \
    --update-freq 8 \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy \
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
    --fp16 \
    --max-update 15000 \
    --log-format simple --log-interval 1 \
    $DATA

    # --share-all-embeddings \






