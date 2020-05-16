DATA=./DATA/data-bin/iwslt14.spm.de-en
TASK=finetune_mixed_2011
MODEL=./naiveNAT/checkpoints/${TASK}/checkpoint_best.pt
USER=./naiveNAT/
RESULT=./result/${TASK}/

fairseq-generate \
    --gen-subset test \
    --task translation_lev_bleu \
    --add-mask-token \
    --path $MODEL \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --iter-decode-max-iter 1 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe sentencepiece \
    --sacrebleu \
    --batch-size 64 \
    --user-dir $USER \
    --results-path $RESULT \
    $DATA

