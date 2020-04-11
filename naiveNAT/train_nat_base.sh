#!/usr/bin/env bash
TASK=nat_base
mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train \
    /hdd/NAT/data-bin/iwslt14.tokenized.de-en/ \
    -s de -t en \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch myNAT \
    --src-embedding-copy \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --noise no_noise \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
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
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 1 \
    --fixed-validation-seed 7 \
    --max-tokens 4096 \
    --save-interval-updates 10000 \
    --max-update 300000 \
    --decoder_positional_attention \
    --fp16 \
    --user-dir .