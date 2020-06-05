#!/usr/bin/env bash
PATH=$HOME/.local/bin:$PATH
WORKERS=8
TASK=mbart_denoise
DATA=../DATA/data-bin/newscrawl.accented
CHECK=./checkpoints/mbart_denoise/checkpoint_best.pt

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train --task nat_multilingual_denoising --user-dir .. \
  --restore-file $CHECK \
  --multilang-sampling-alpha 0.1 \
  --langs en,de,es,fr \
  --add-lang-token \
  --decoder-input-noise random_mask \
  --sample-break-mode eos \
  --mask 0.35 \
  --mask-random 0.0 \
  --mask-length span-poisson \
  --poisson-lambda 3.5 \
  --replace-length 1 \
  --insert 0.0 \
  --rotate 0.0 \
  --permute 0.0 \
  --permute-sentences 0.0 \
  --max-tokens 4096 \
  --max-tokens-valid 512 \
  --update-freq 32 \
  --tokens-per-sample 512 \
  --arch nonautoregressive_transformer \
  --share-all-embeddings \
  --apply-bert-init \
  --pred-length-offset \
  --length-loss-factor 0.1 \
  --sg-length-pred \
  --criterion nat_loss \
  --label-smoothing 0.1 \
  --optimizer adam \
  --adam-betas '(0.9, 0.999)' \
  --adam-eps 1e-6 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --lr 5e-4 \
  --clip-norm 0.0 \
  --dropout 0.3 \
  --weight-decay 0.01 \
  --no-epoch-checkpoints \
  --best-checkpoint-metric ppl \
  --save-dir checkpoints/$TASK \
  --tensorboard-logdir logdir/$TASK \
  --num-workers $WORKERS \
  --max-update 50000 \
  --memory-efficient-fp16 \
  --log-format tqdm --log-interval 1 \
  --save-interval-updates 150 --keep-interval-updates 1 \
  --skip-invalid-size-inputs-valid-test \
  $DATA

  # --memory-efficient-fp16
  # --eval-lm-detok moses \
  # --eval-lm-remove-bpe \
  # --eval-lm-print-samples \

  # --tokens-per-sample 512 \
  # --add-lang-token (this has bug for now !

  # sample-break-mode {none, complete, complete_doc, eos}: 
  #   none: fills each sample with tokens-per-sample tokens
  #   complete: splits samples only at the end of sentence, but may include multiple sentences per sample. 
  #   complete_doc: similar but respects doc boundaries.
  #   eos: only one sentence per sample.

  # --truncate-sequence

  # mask: fraction of words/subwords that will be masked
  # mask-random: instead of using [MASK], use random token this often

  # insert: insert this percentage of additional random tokens
  # rotate: rotate this proportion of inputs  
  # permute-sentences: shuffle this proportion of sentences in all inputs

  # mask-length: ('subword', 'word', 'span-poisson')
  # poisson-lambda: lambda for above span-poisson
  # replace-length: when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)

  # permute: take this proportion of subwords and permute them (looks like this is not used ?)
  






