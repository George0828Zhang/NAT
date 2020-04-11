# Preprocessing For Pretraining

1. Get iwslt-2014 data:
```bash
FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
./get_iwslt_data.sh
``` 

2. Get wikitext-103 data, and process with vocabulary created above
```bash
FAIRSEQDIR=$HOME/Projects/NATLab/fairseq
./get_wikitext_data.sh
```