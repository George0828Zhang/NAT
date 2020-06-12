#!/usr/bin/env bash
PATH=$HOME/.local/bin:$PATH
TASK=leanat
DATA=../DATA/data-bin/wmt14.accented
CHECK=../naiveNAT/checkpoints/teacher_wmt14/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
    # --control-vae --control-vae-args '{"v_kl": 3.0, "Kp": 0.01, "Ki": 0.0001, "beta_min": 0.0, "beta_max": 1.0 }' \
    # --restore-file $CHECK \
    # --reset-optimizer --reset-dataloader --reset-meters \
    # --load-weight-level encoder_decoder \    
    # --use-mask-token \

fairseq-train --task translation_lev_bleu --user-dir .. \
    --noise no_noise \
    -s de -t en \
    --restore-file $CHECK \
    --reset-optimizer --reset-dataloader --reset-meters \
    --load-weight-level encoder \
    --max-tokens 4096 \
    --max-tokens-valid 1024 \
    --empty-cache-freq 1 \
    --update-freq 25 \
    --arch lea_nat \
    --share-all-embeddings \
    --apply-bert-init \
    --pred-length-offset \
    --length-loss-factor 1.0 \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --lr 2.5e-4 \
    --clip-norm 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 1}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 8 \
    --memory-efficient-fp16 \
    --fp16-scale-tolerance 3 \
    --min-loss-scale 1e-8 \
    --max-update 50000 \
    --save-interval-updates 500 --keep-interval-updates 1 \
    --log-format tqdm --log-interval 1 \
    $DATA
