# Improving Learned Index Structures

## DeeperImpact: Optimizing Sparse Learned Index Structures

[![arXiv](https://img.shields.io/badge/arXiv-2405.17093-green.svg)](https://arxiv.org/abs/2405.17093) [![Static Badge](https://img.shields.io/badge/HuggingFace-DeeperImpact-blue)](https://huggingface.co/soyuj/deeper-impact) [![Static Badge](https://img.shields.io/badge/HuggingFace-Llama2_doc2query-blue)](https://huggingface.co/soyuj/llama2-doc2query)

A lot of recent work has focused on sparse learned indexes that use deep neural architectures to significantly improve
retrieval quality while keeping the efficiency benefits of the inverted index. While such sparse learned structures
achieve effectiveness far beyond those of traditional inverted index-based rankers, there is still a gap in
effectiveness to the best dense retrievers, or even to sparse methods that leverage more expensive optimizations such as
query expansion and query term weighting. We focus on narrowing this gap by revisiting and optimizing DeepImpact, a
sparse retrieval approach that uses DocT5Query for document expansion followed by a BERT language model to learn impact
scores for document terms. We first reinvestigate the expansion process and find that the recently proposed Doc2Query--
query filtration does not enhance retrieval quality when used with DeepImpact. Instead, substituting T5 with a
fine-tuned Llama 2 model for query prediction results in a considerable improvement. Subsequently, we study training
strategies that have proven effective for other models, in particular the use of hard negatives, distillation, and
pre-trained CoCondenser model initialization. Our results significantly narrow the effectiveness gap with the most
effective versions of SPLADE.

## Installation :computer:

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage :hammer_and_wrench:

Checkout the [notebook](./inference_deeper_impact.ipynb) for a detailed example on how to use DeeperImpact, from
expansions to running inference.

### Expansions

To run expansions on a collection of documents, use the following command:

```bash
python -m src.llama2.generate \
    --llama_path <path | HuggingFaceHub link> \
    --collection_path <path> \
    --collection_type [msmarco | beir] \
    --output_path <path> \
    --batch_size <batch_size> \
    --max_tokens 512 \
    --num_return_sequences 80 \
    --max_new_tokens 50 \
    --top_k 50 \
    --top_p 0.95 \
    --peft_path soyuj/llama2-doc2query
```

This will generate a jsonl file with expansions for each document in the collection. To append the unique expansion
terms to the original collection, use the following command:

```bash
python -m src.llama2.merge \
  --collection_path <path> \
  --collection_type [msmarco | beir] \
  --queries_path <jsonl file generated above> \
  --output_path <path>
```

### Training

To train DeeperImpact, use the following command:

```bash
torchrun --standalone --nproc_per_node=gpu -m src.deep_impact.train \
  --dataset_path <cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz> \
  --queries_path <queries.train.tsv> \
  --collection_path <expanded_collection_path> \
  --checkpoint_dir <checkpoint_dir_path> \
  --batch_size <batch_size> \
  --save_every <n> \
  
  # optional if you want to start with a checkpoint
  --start_with <specific_checkpoint_path> \
  
  # if you want to use distillation
  # instead, if you want to use triples and cross-entropy, pass triples in dataset_path and exclude this flag
  --distil_kl \
  
  # maximum token length for each document
  --max_length 300 \
  
  --lr 1e-6 \
  --seed 42 \
  --gradient_accumulation_steps 1 \
  
  # experimental options
  
  # for triples and cross-entropy with in-batch negatives
  # pass triples in dataset_path and exclude --distil_kl
  # --in_batch_negatives \
  
  # for distillation using MarginMSELoss instead of KL divergence loss
  # pass the same cross-encoder dataset, exclude --distil_kl, and include the qrels_path
  # --distil_mse \
  # --qrels_path qrels.train.tsv \
  
  # to train cross-encoder DeepImpact model, pass triples and exclude --distil_kl
  # --cross_encoder 
```

It distributes the training across multiple GPUs in the machine. The `batch_size` is per GPU. To manually set the GPUs,
use `CUDA_VISIBLE_DEVICES` environment variable.

### Inference

To run inference on a collection of documents, use the following command:

```bash
python -m src.deep_impact.index \
  --collection_path <expanded_collection.tsv> \
  --output_file_path <path> \
  --model_checkpoint_path <model_checkpoint_path> \
  --num_processes <n> \
  --process_batch_size <process_batch_size> \
  --model_batch_size <model_batch_size>
```

It distributes the inference across multiple GPUs in the machine. To manually set the GPUs, use `CUDA_VISIBLE_DEVICES`
environment variable.

### Quantization

To quantize the generated impact scores, use the following command:

```bash
python -m src.deep_impact.quantize \
  -i <deep_impact_collection_path> \
  -o <quantized_deep_impact_collection_path>
```

You can then use Anserini to generate the inverted index and export it in CIFF format, which can then be directly
processed with PISA.

---

> For quick experimentation, you can also use a custom implementation of an inverted index:
>
> ```bash
> python -m src.deep_impact.inverted_index.create \
>   -i <quantized_deepimpact_collection_path> \
>   -o <inverted_index_dir_path>
> ```
> 
> To rank:
> 
> ```bash
> python -m src.deep_impact.rank \
>   --index_path <inverted_index_dir_path> \
>   --queries_path <queries_to_rank> \
>   --output_path <run_file_path> \
>   --dataset_type [msmarco | beir] \
>   --num_workers <n>
> ```
> 
> To evaluate:
> 
> ```bash
> python -m src.deep_impact.evaluate \
>   --run_file_path <run_file_path> \
>   --qrels_path <qrels_path>
> ```

---

## Contact :email:

For any questions or comments, please reach out to us via email: soyuj@nyu.edu

## Citation :bookmark_tabs:

Please cite our work as:

* DeeperImpact: Optimizing Sparse Learned Index Structures

```
@misc{basnet2024deeperimpact,
      title={DeeperImpact: Optimizing Sparse Learned Index Structures}, 
      author={Soyuj Basnet and Jerry Gou and Antonio Mallia and Torsten Suel},
      year={2024},
      eprint={2405.17093},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## Resources

### Transformers
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([paper](https://arxiv.org/pdf/1810.04805.pdf))

### Document Expansion
- Document Expansion by Query Prediction ([paper](https://arxiv.org/pdf/1904.08375.pdf))
- From doc2query to docTTTTTquery ([paper](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf))
- Doc2Query--: When Less is More ([paper](https://arxiv.org/pdf/2301.03266.pdf))

### Information Retrieval Models
- Passage Re-ranking with BERT ([paper](https://arxiv.org/pdf/1901.04085.pdf))
- Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval ([paper](https://arxiv.org/pdf/1910.10687.pdf))
- Context-Aware Document Term Weighting for Ad-Hoc Search ([paper](https://www.cs.cmu.edu/~callan/Papers/TheWebConf20-Zhuyun-Dai.pdf))
- Efficiency Implications of Term Weighting for Passage Retrieval ([paper](https://www.cs.cmu.edu/~zhuyund/papers/SIGIR2020DeepCT-efficiency.pdf))
- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT ([paper](https://arxiv.org/pdf/2004.12832.pdf))
- SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking ([paper](https://arxiv.org/pdf/2107.05720.pdf))
- Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation ([paper](https://arxiv.org/pdf/2110.11540.pdf))

### DeepImpact
- **Original Paper**: Learning Passage Impacts for Inverted Indexes ([paper](https://arxiv.org/pdf/2104.12016.pdf))
- Faster Learned Sparse Retrieval with Guided Traversal ([paper](https://arxiv.org/pdf/2204.11314.pdf))
- A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques ([paper](https://arxiv.org/pdf/2106.14807.pdf))

### Text REtrieval Conference (TREC)
- Overview of the TREC 2019 Deep Learning Track ([paper](https://arxiv.org/pdf/2003.07820.pdf))
- Overview of the TREC 2020 Deep Learning Track ([paper](https://arxiv.org/pdf/2102.07662.pdf))
- Overview of the TREC 2021 Deep Learning Track ([paper](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/trec2021-deeplearning-overview.pdf))