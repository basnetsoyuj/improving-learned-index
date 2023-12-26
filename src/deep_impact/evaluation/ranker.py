from pathlib import Path
from typing import Union

from src.deep_impact.evaluation.metrics import Metrics
from src.deep_impact.inverted_index import InvertedIndex
from src.deep_impact.models import DeepImpact as Model
from src.utils.datasets import QueryRelevanceDataset, Queries
import time

class Ranker:
    def __init__(
            self,
            index_path: Union[str, Path],
            qrels_path: Union[str, Path],
            queries_path: Union[str, Path],
            output_path: Union[str, Path],
    ):
        self.index = InvertedIndex(index_path=index_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.queries = Queries(queries_path=queries_path)
        self.metrics = Metrics(mrr_depths=[10], recall_depths=[10, 20, 200, 1000])
        self.output_path = Path(output_path)

    def run(self):
        avg_time = 0
        for i, qid in enumerate(self.qrels.keys(), start=1):
            start = time.time()
            self.rank(qid=qid)
            end = time.time()
            avg_time += end - start
            print(f'Average time for query {i}: {avg_time / i}')

            # rankings = self.rank(qid=qid)
            # with open(self.output_path, 'a') as f:
            #     for pid, score in rankings:
            #         f.write(f'{qid}\t{pid}\t{score}\n')

    def rank(self, qid: int):
        query_terms = Model.process_query(query=self.queries[qid])
        scores = self.index.score(query_terms=query_terms)
        self.metrics.add_result(qid=qid, rankings=scores, gold_positives=self.qrels[qid])
        self.metrics.log_metrics()
        return scores
