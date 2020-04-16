# Preprocessing For Pretraining

1. Get monolingual data from newscrawl 2017:
```bash
export FAIRSEQDIR=/your/path/to/fairseq
bash get_newscrawl_mono_en_de.sh
```
2. Get iwslt-2014 data and preprocess using dict above:
```bash
export FAIRSEQDIR=/your/path/to/fairseq
export NEWSDIR=data-bin/news.2017.en-de
bash get_iwslt_data_and_used_dict.sh
```