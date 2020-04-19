fairseq-generate \
    ./DATA/data-bin/iwslt14.tokenized.de-en/ \
    --gen-subset test \
    --task translation_lev \
    --path naiveNAT/checkpoints/gu/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 32 \
    --user-dir ./naiveNAT