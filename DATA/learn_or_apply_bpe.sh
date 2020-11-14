#!/usr/bin/env bash
BPEROOT=subword-nmt/subword_nmt
if [[ ! -d $BPEROOT ]]; then
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
fi


# SPM
if [[ -z "$SPM_MODEL" ]]; then
    echo "Don't apply SPM."
else
    if [[ "$SPM_MODEL" == "Current" ]]; then
        echo "Didn't provide SPM model, learn on current dataset"
        SPM_PREFIX=$CACHE/prep/spm
        SPM_MODEL=$SPM_PREFIX.model
        BPE_TRAIN=$CACHE/prep/all.bpe-train

        if [ -f $SPM_MODEL ]; then
            echo "SPM model: $SPM_MODEL exists, skip learning"
        else
            if [ -f $BPE_TRAIN ]; then
                echo "$BPE_TRAIN found, skipping concat."
            else
                for l in $SRCLANG $TGTLANG; do \
                    train=$CACHE/prep/train.$l
                    valid=$CACHE/prep/valid.$l
                    default=1000000
                    total=$(cat $train $valid | wc -l)
                    echo "lang $l total: $total."
                    if [ "$total" -gt "$default" ]; then
                        cat $train $valid | \
                        shuf -r -n $default >> $BPE_TRAIN
                    else
                        cat $train $valid >> $BPE_TRAIN
                    fi                    
                done
            fi

            echo "spm_train on ${BPE_TRAIN}..."
            ccvg=1.0
            if [[ $SRCLANG == "zh" ]] || [[ $TGTLANG == "zh" ]]; then
                ccvg=0.9995
            fi
            spm_train --input=$BPE_TRAIN \
                --model_prefix=$SPM_PREFIX \
                --vocab_size=$BPE_TOKENS \
                --character_coverage=$ccvg \
                --model_type=unigram \
                --normalization_rule_name=nmt_nfkc_cf

            mkdir -p $OUTDIR
            cp $SPM_MODEL $OUTDIR/spm.model
            #######################################################
        fi

    else
        if [[ ! -f $SPM_MODEL ]]; then
            echo "SPM model: $SPM_MODEL not found!"
            exit
        fi
    fi

    echo "Using SPM model $SPM_MODEL"
    for L in $SRCLANG $TGTLANG; do
        for f in train.$L valid.$L test.$L; do
            prep=$CACHE/prep
            ready=$CACHE/ready

            if [ -f $ready/$f ]; then
                echo "found $ready/$f, skipping spm_encode"
            else
                echo "spm_encode to ${f}..."
                spm_encode --model=$SPM_MODEL \
                    --output_format=piece \
                    < $prep/$f > $ready/$f
            fi
        done
    done
fi

# BPE
if [[ -z "$BPE_CODE" ]]; then
    echo "Don't apply BPE."
else
    if [[ "$BPE_CODE" == "Current" ]]; then
        echo "Didn't provide BPE code, learn on current dataset"
        BPE_CODE=$CACHE/prep/code
        BPE_TRAIN=$CACHE/prep/all.bpe-train

        if [ -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE exists, skip learning"
        else
            if [ -f $BPE_TRAIN ]; then
                echo "$BPE_TRAIN found, skipping concat."
            else
                for l in $SRCLANG $TGTLANG; do \
                    train=$CACHE/prep/train.$l
                    valid=$CACHE/prep/valid.$l
                    default=1000000
                    total=$(cat $train $valid | wc -l)
                    echo "lang $l total: $total."
                    if [ "$total" -gt "$default" ]; then
                        cat $train $valid | \
                        shuf -r -n $default >> $BPE_TRAIN
                    else
                        cat $train $valid >> $BPE_TRAIN
                    fi                    
                done
            fi

            echo "learn_bpe.py on $BPE_TRAIN..."
            python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $BPE_TRAIN > $BPE_CODE
            mkdir -p $OUTDIR
            cp $BPE_CODE $OUTDIR/code
            #######################################################
        fi

    else

        if [ ! -f $BPE_CODE ]; then
            echo "BPE CODE: $BPE_CODE not found!"
            exit
        fi

    fi

    echo "use bpe code at $BPE_CODE"

    for L in $SRCLANG $TGTLANG; do
        for f in train.$L valid.$L test.$L; do
            prep=$CACHE/prep
            ready=$CACHE/ready

            if [ -f $ready/$f ]; then
                echo "found $ready/$f, skipping apply_bpe"
            else
                echo "apply_bpe.py to ${f}..."
                python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $ready/$f
            fi
        done
    done

fi