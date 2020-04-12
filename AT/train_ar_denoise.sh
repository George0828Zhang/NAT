#!/usr/bin/env bash
TASK=ar_denoise2
DATA=../DATA/data-bin/news.2017.en-de
mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

fairseq-train --task multilingual_denoising \
  --multilang-sampling-alpha 0.7 \
  --langs de,en \
  --sample-break-mode complete_doc \
  --mask 0.15 \
  --mask-random 0.3 \
  --mask-length span-poisson \
  --poisson-lambda 3.0 \
  --replace-length 1 \
  --insert 0.1 \
  --rotate 0.1 \
  --permute 0.1 \
  --permute-sentences 0.5 \
  --max-tokens 4096 \
  --update-freq 16 \
  --tokens-per-sample 512 \
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
  --dropout 0.1 \
  --weight-decay 0.0001 \
  --no-epoch-checkpoints \
  --best-checkpoint-metric ppl \
  --save-dir checkpoints/$TASK \
  --tensorboard-logdir logdir/$TASK \
  --num-workers 4 \
  --fp16 \
  --max-update 200000 \
  --log-format simple --log-interval 1 \
  --save-interval-updates 150 --keep-interval-updates 2 \
  --skip-invalid-size-inputs-valid-test \
  $DATA

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
  







