from __future__ import annotations

import logging
from typing import Literal
from collections import defaultdict
import heapq

from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
import torch
from tqdm import tqdm

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]


MAPPING_DATASET_NAME_TO_ID = {
    "climatefever": "zeta-alpha-ai/NanoClimateFEVER",
    "dbpedia": "zeta-alpha-ai/NanoDBPedia",
    "fever": "zeta-alpha-ai/NanoFEVER",
    "fiqa2018": "zeta-alpha-ai/NanoFiQA2018",
    "hotpotqa": "zeta-alpha-ai/NanoHotpotQA",
    "msmarco": "zeta-alpha-ai/NanoMSMARCO",
    "nfcorpus": "zeta-alpha-ai/NanoNFCorpus",
    "nq": "zeta-alpha-ai/NanoNQ",
    "quoraretrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
    "scidocs": "zeta-alpha-ai/NanoSCIDOCS",
    "arguana": "zeta-alpha-ai/NanoArguAna",
    "scifact": "zeta-alpha-ai/NanoSciFact",
    "touche2020": "zeta-alpha-ai/NanoTouche2020",
}

MAPPING_DATASET_NAME_TO_HUMAN_READABLE = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}

class Dataset:
    def __init__(self, queries, corpus, relevant_docs, name):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.name = name


class SparseSearch:
    def __init__(self, model, batch_size, verbose=False):
        self.model = model
        self.batch_size = batch_size
        self.inverted_index = defaultdict(list)  # term_id -> [(doc_id, score), ...]
        self.corpus_ids = []
        self.verbose = verbose
        
    def _build_inverted_index(self, corpus):
        """Build inverted index from corpus embeddings"""
        if self.verbose:
            print(f"Building inverted index for {len(corpus)} documents...")
        
        corpus_ids = list(corpus.keys())
        self.corpus_ids = corpus_ids
        
        # Process corpus in batches
        iterator = tqdm(range(0, len(corpus), self.batch_size), desc="Building inverted index") if self.verbose else range(0, len(corpus), self.batch_size)
        for i in iterator:
            batch_texts = list(corpus.values())[i:i+self.batch_size]
            batch_ids = corpus_ids[i:i+self.batch_size]
            
            with torch.no_grad():
                embeddings = self.model.get_impact_scores_batch(batch_texts)
            
            # Process each document's embedding
            for doc_id, embedding in zip(batch_ids, embeddings):
                for term_id, score in embedding:
                    if score > 0:  # Only store non-zero scores
                        self.inverted_index[term_id].append((doc_id, score))        
        if self.verbose:
            print(f"Built inverted index with {len(self.inverted_index)} terms")
        
    def search(self, queries, corpus, k):
        # Build inverted index if not already built
        if not self.inverted_index:
            self._build_inverted_index(corpus)
        
        results = {}
        if self.verbose:
            print(f"Searching for {len(queries)} queries...")
        
        iterator = tqdm(queries.items(), desc="Searching queries") if self.verbose else queries.items()
        for query_id, query in iterator:
            query_terms = self.model.process_query(query)
            
            doc_scores = defaultdict(float)            
            # Score documents using inverted index
            for query_term in query_terms:
                if query_term in self.inverted_index:
                    for doc_id, doc_score in self.inverted_index[query_term]:
                        doc_scores[doc_id] += doc_score  # Impact score multiplication
            
            # Get top-k documents for this query
            if len(doc_scores) == 0:
                results[query_id] = {}
            else:
                # Use heapq to efficiently get top-k
                if k >= len(doc_scores):
                    top_k_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                else:
                    top_k_docs = heapq.nlargest(k, doc_scores.items(), key=lambda x: x[1])
                
                results[query_id] = {doc_id: float(score) for doc_id, score in top_k_docs}
        
        if self.verbose:
            print(f"Retrieved top-{k} documents for {len(queries)} queries")
        return results

class BaseEvaluator:
    def __init__(self, batch_size=16, verbose=False):
        self.verbose = verbose
        self.batch_size = batch_size
        
    def _load_dataset(self, dataset_name: DatasetNameType) -> Dataset:
        pass
    
    def evaluate_dataset(self, model, dataset_name):
        pass
    
    def evaluate_all(self, model):
        pass

class NanoBEIREvaluator(BaseEvaluator):
    def __init__(self, batch_size=16, verbose=False):
        super().__init__(batch_size, verbose)
        
    def _load_dataset(
        self, dataset_name: DatasetNameType,
    ) -> Dataset:

        if self.verbose:
            print(f"Loading dataset {dataset_name}...")

        dataset_path = MAPPING_DATASET_NAME_TO_ID[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = {}
            qrels_dict[sample["query-id"]][sample["corpus-id"]] = 1

        human_readable_name = MAPPING_DATASET_NAME_TO_HUMAN_READABLE[dataset_name]
        return Dataset(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
        )
        
    def evaluate_all(self, model):
        metrics = {}
        for dataset_name in MAPPING_DATASET_NAME_TO_ID.keys():
            if self.verbose:
                print(f"Evaluating dataset {dataset_name}...")
            metrics[dataset_name] = self.evaluate_dataset(model, dataset_name)
            if self.verbose:
                print(f"Metrics for {dataset_name}: {metrics[dataset_name]}")
        #compute average metrics and add it
        avg_metrics = (
            {
                'NDCG@10': sum(metrics[dataset_name][0]['NDCG@10'] for dataset_name in metrics) / len(metrics),
                'NDCG@100': sum(metrics[dataset_name][0]['NDCG@100'] for dataset_name in metrics) / len(metrics),
                'NDCG@1000': sum(metrics[dataset_name][0]['NDCG@1000'] for dataset_name in metrics) / len(metrics)
            },
            {
                'MAP@10': sum(metrics[dataset_name][1]['MAP@10'] for dataset_name in metrics) / len(metrics),
                'MAP@100': sum(metrics[dataset_name][1]['MAP@100'] for dataset_name in metrics) / len(metrics),
                'MAP@1000': sum(metrics[dataset_name][1]['MAP@1000'] for dataset_name in metrics) / len(metrics)
            },
            {
                'Recall@10': sum(metrics[dataset_name][2]['Recall@10'] for dataset_name in metrics) / len(metrics),
                'Recall@100': sum(metrics[dataset_name][2]['Recall@100'] for dataset_name in metrics) / len(metrics),
                'Recall@1000': sum(metrics[dataset_name][2]['Recall@1000'] for dataset_name in metrics) / len(metrics)
            },
            {
                'P@10': sum(metrics[dataset_name][3]['P@10'] for dataset_name in metrics) / len(metrics),
                'P@100': sum(metrics[dataset_name][3]['P@100'] for dataset_name in metrics) / len(metrics),
                'P@1000': sum(metrics[dataset_name][3]['P@1000'] for dataset_name in metrics) / len(metrics)
            }
        )
        metrics["avg"] = avg_metrics
        return metrics
            
    def evaluate_dataset(self, model, dataset_name):
        dataset = self._load_dataset(dataset_name)
        searcher = SparseSearch(model, batch_size=self.batch_size, verbose=self.verbose)
        results = searcher.search(dataset.queries, dataset.corpus, k=1000)
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(dataset.relevant_docs, results, [10, 100, 1000])
        return metrics
            
    
    
if __name__ == "__main__":
    from src.deep_impact.models.original import DeepImpact
    model = DeepImpact.load('soyuj/deeper-impact')
    model.to('cuda')
    model.eval()
    evaluator = NanoBEIREvaluator(verbose=True, batch_size=16)
    metrics = evaluator.evaluate_all(model)
    print(metrics)
    
            