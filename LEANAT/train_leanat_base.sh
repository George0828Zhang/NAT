#!/usr/bin/env bash
TASK=lea_nat_scratch
DATA=../DATA/data-bin/iwslt16.lee
CHECK=../naiveNAT/checkpoints/teacher/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
    
fairseq-train --task translation_lev_bleu --user-dir . \
    --noise random_mask \
    -s en -t de \
    --load-encoder-only \
    --restore-file $CHECK \
    --reset-optimizer --reset-dataloader --reset-meters \
    --max-tokens 4096 \
    --max-tokens-valid 2048 \
    --update-freq 12 \
    --arch lea_iwslt_16 \
    --share-all-embeddings \
    --apply-bert-init \
    --lea-loss-factor 1.0 \
    --lea-attention-heads 2 \
    --lea-use-embed \
    --sg-lea-pred \
    --sg-length-pred \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --criterion nat_loss \
    --label-smoothing 0.0 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 746 \
    --lr 5e-4 \
    --clip-norm 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 4 \
    --max-update 100000 \
    --fp16 \
    --log-format simple --log-interval 1 \
    $DATA    


    # --eval-bleu-detok space \
    # --eval-bleu-remove-bpe sentencepiece \
    # --eval-bleu-print-samples \
    # --save-interval-updates 500 --keep-interval-updates 5 \
    # --skip-invalid-size-inputs-valid-test \
    # --fp16 \





