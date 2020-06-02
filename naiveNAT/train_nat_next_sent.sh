#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
TASK=next_sent_denoise
DATA=../DATA/data-bin/europarl
mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

# cs,de,en,es,fi,fr,lt,pl,pt
fairseq-train --task nat_next_sentence_generation --user-dir .. \
  --multilang-sampling-alpha 0.1 \
  --add-lang-token \
  --langs cs,de,en,es,fi,fr,lt,pl,pt \
  --sample-break-mode eos \
  --randomize-mask-ratio \
  --mask-random 0.0 \
  --mask-length word \
  --poisson-lambda 3.5 \
  --replace-length -1 \
  --insert 0.0 \
  --rotate 0.0 \
  --permute 0.0 \
  --permute-sentences 0.0 \
  --max-tokens 4096 \
  --max-tokens-valid 1024 \
  --update-freq 25 \
  --tokens-per-sample 128 \
  --arch gu \
  --share-all-embeddings \
  --apply-bert-init \
  --src-embedding-copy \
  --length-loss-factor 0.0 \
  --sg-length-pred \
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
  --weight-decay 0.01 \
  --no-epoch-checkpoints \
  --eval-lm-remove-bpe \
  --eval-lm-detok moses \
  --eval-lm-print-samples \
  --best-checkpoint-metric ppl \
  --save-dir checkpoints/$TASK \
  --tensorboard-logdir logdir/$TASK \
  --num-workers 4 \
  --fp16 \
  --max-update 200000 \
  --log-format simple --log-interval 1 \
  --save-interval-updates 5 --keep-interval-updates 2 \
  --skip-invalid-size-inputs-valid-test \
  $DATA

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
  







