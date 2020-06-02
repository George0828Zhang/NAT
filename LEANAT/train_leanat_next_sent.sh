#!/usr/bin/env bash
# PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=3
TASK=lea_denoise
DATA=../DATA/data-bin/newscrawl.accented

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK
    
fairseq-train --task nat_next_sentence_generation --user-dir ../ \
    --multilang-sampling-alpha 0.1 \
    --langs fr,es \
    --add-lang-token \
    --sample-break-mode eos \
    --mask 0.35 \
    --mask-random 0.3 \
    --mask-length word \
    --replace-length -1 \
    --insert 0.0 \
    --rotate 0.0 \
    --permute 0.0 \
    --permute-sentences 0.0 \
    --max-tokens 8192 \
    --max-tokens-valid 1024 \
    --update-freq 25 \
    --tokens-per-sample 128 \
    --arch lea_nat \
    --lea-loss-factor 1.0 \
    --lea-attention-heads 4 \
    --sg-lea-pred \
    --lea-use-embed \
    --share-decoder-input-output-embed \
    --apply-bert-init \
    --length-loss-factor 0.0 \
    --sg-length-pred \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --lr 5e-4 \
    --clip-norm 0.0 \
    --dropout 0.3 \
    --weight-decay 0.01 \
    --no-epoch-checkpoints \
    --eval-lm-detok moses \
    --eval-lm-remove-bpe \
    --eval-lm-print-samples \
    --best-checkpoint-metric ppl \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --num-workers 4 \
    --fp16 \
    --max-update 200000 \
    --log-format simple --log-interval 1 \
    --save-interval-updates 500 --keep-interval-updates 10 \
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
    






