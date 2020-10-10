# Preprocessing Usage

1. Modify `get_data_final.sh` to suit your need
```bash
export PREFIX=/media/george/Data    # path to cache raw data 
export DATABIN=$(pwd)/data-bin      # path to output data
export CONFIG=config/multi30k       # path to config for dataset, including {download, preprocess, path}.sh
export SRCLANG=de                   # source language
export TGTLANG=en                   # source language
export WORKERS=4                    # workers
export BPE_TOKENS=4000              # desired bpe tokens or spm pieces. only used if path.sh specify 'Current' (i.e. to learn bpe)
#export DISTILL='true'              # optionally specify to use distillation data. 
```

2. Optionally create 'config/new_dataset'
- The folder needs to include these files
```bash
download.sh
path.sh
preprocess.sh
```
- Set the config in `get_data_final.sh`
```bash
export CONFIG=config/new_dataset
```