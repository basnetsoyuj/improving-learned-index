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

---

### Convert Checkpoint

To download the LLaMa model, follow the guide [here](https://github.com/facebookresearch/llama).

We need to convert the checkpoint from its original format into the dedicated Hugging Face format. Then run:

```bash
pip install -r requirements.txt
```

Assuming your downloaded checkpoint resides under `models/7B`, run the following:

```bash
TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
python ${TRANSFORM} --input_dir models --model_size 7B --output_dir models_hf/7B
```

Then, you can use the converted checkpoint to fine-tune LLaMa 2 on MS MARCO dataset:

```bash
python -m src.llama2.finetune --enable_profiler \
  --checkpoint_dir /hdd1/home/soyuj/llama2/models_hf/7B \
  --dataset_path /hdd1/home/soyuj/llama2/document-query-pairs.train.tsv \
  --output_dir /hdd1/home/soyuj/llama2/7B_finetuned \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --lr 1e-5 \
  --epochs 1
```

