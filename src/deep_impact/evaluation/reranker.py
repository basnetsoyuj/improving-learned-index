import multiprocessing
from pathlib import Path
from typing import Union, List, Tuple

import torch
from torch.nn import DataParallel
from tqdm.auto import tqdm

from src.deep_impact.models import DeepImpact
from src.utils.datasets import Collection, TopKRunFile, Queries, RunFile


class ReRanker:
    def __init__(
            self,
            checkpoint_path: Union[str, Path],
            top_k_run_file_path: Union[str, Path],
            queries_path: Union[str, Path],
            collection_path: Union[str, Path],
            output_path: Union[str, Path],
            batch_size: int = 128,
            num_processes: int = 4,
    ):
        self.device = torch.device("cuda")
        self.top_k = TopKRunFile(run_file_path=top_k_run_file_path)
        self.queries = Queries(queries_path=queries_path)
        self.collection = Collection(collection_path=collection_path)
        self.batch_size = batch_size

        self.model = DeepImpact.load(checkpoint_path=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)

        self.run_file = RunFile(run_file_path=output_path)

        self.pool = multiprocessing.Pool(processes=num_processes)
        self.cache = {}

    def run(self):
        with tqdm(total=len(self.top_k)) as pbar:
            for qid, pids in self.top_k:
                scores = self.rerank(qid, pids, pbar=pbar)
                self.run_file.writelines(qid, scores)
                pbar.update(1)

    def save(self, pids, batch_doc_term_scores: List[List[Tuple[str, float]]]):
        for pid, doc_term_scores in zip(pids, batch_doc_term_scores):
            self.cache[pid] = {term: score for term, score in doc_term_scores}

    def score(self, pid, query_terms):
        return sum(self.cache[pid].get(term, 0) for term in query_terms)

    @torch.no_grad()
    def rerank(self, qid: int, pids, pbar):
        query_terms = DeepImpact.process_query(query=self.queries[qid])

        batch = []
        batch_pids = []
        scores = []
        for i, pid in enumerate(pids):
            pbar.set_description(f"Reranking query {qid} ({i + 1}/{len(pids)})")

            if pid in self.cache:
                scores.append(self.score(pid, query_terms))
                continue

            batch.append(self.collection[pid])
            batch_pids.append(pid)

            if len(batch) == self.batch_size or (i == len(pids) - 1 and batch):
                batch_encoded, batch_term_to_token_index_map = zip(
                    *list(self.pool.map(DeepImpact.process_document, batch))
                )
                input_ids = torch.tensor([x.ids for x in batch_encoded], dtype=torch.long)
                attention_mask = torch.tensor([x.attention_mask for x in batch_encoded], dtype=torch.long)
                type_ids = torch.tensor([x.type_ids for x in batch_encoded], dtype=torch.long)

                batch_term_impacts = DeepImpact.compute_term_impacts(
                    documents_term_to_token_index_map=batch_term_to_token_index_map,
                    outputs=self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids)
                )

                self.save(batch_pids, batch_term_impacts)
                scores.extend([self.score(pid, query_terms) for pid in batch_pids])

                batch = []
                batch_pids = []

        return sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)[:1000]
