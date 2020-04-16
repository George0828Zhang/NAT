#!/usr/bin/env bash
TASK=nat_finetune
DATA=../DATA/data-bin/iwslt14.tokenized.de-en
CHECK=checkpoints/nat_denoise_short/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train --task translation_lev_bleu --user-dir . \
    --noise no_noise \
    --add-mask-token \
    -s de -t en \
    --restore-file $CHECK \
    --reset-optimizer --reset-dataloader --reset-meters \
    --max-tokens 3000 \
    --max-tokens-valid 1000 \
    --empty-cache-freq 1 \
    --update-freq 8 \
    --arch nonautoregressive_transformer \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --share-all-embeddings \
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
    --fp16 \
    --max-update 15000 \
    --log-format simple --log-interval 1 \
    $DATA

    # --share-all-embeddings \






