## Fine-tuning LLaMa 2

This directory contains the code for fine-tuning LLaMa 2 on MS MARCO dataset's document-query pairs.

### Requirements

* Download the query id-relevant document id
  pairs ([qrels.train.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv)) from MS MARCO Dataset.

  Also download [collection.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz)
  and [queries.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz) that contain the actual
  documents and queries respectively.

* Download
  the [pre-trained LLaMa 2 model](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
  from Meta's Website.
  You can request access for the weights [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).


### Setup

After downloading the datasets, run `prepare_dataset.py` to create query-document pair dataset:

```bash
python -m src.llama2.prepare_dataset \
  --qrels_path qrels.train.tsv \
  --queries_path queries.train.tsv \
  --collection_path collection.tsv \
  --output_path document-query-pairs.tsv
```