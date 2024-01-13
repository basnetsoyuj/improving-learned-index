import time
from itertools import product
from pathlib import Path
from typing import Union

from src.deep_impact.evaluation.metrics import Metrics
from src.deep_impact.inverted_index import InvertedIndex
from src.deep_impact.models import DeepImpact, DeepPairwiseImpact
from src.utils.datasets import QueryRelevanceDataset, Queries


class Ranker:
    def __init__(
            self,
            index_path: Union[str, Path],
            qrels_path: Union[str, Path],
            queries_path: Union[str, Path],
            output_path: Union[str, Path],
            pairwise: bool = False,
    ):
        self.index = InvertedIndex(index_path=index_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.queries = Queries(queries_path=queries_path)
        self.metrics = Metrics(
            mrr_depths=[10],
            recall_depths=[10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        )
        self.output_path = Path(output_path)
        self.pairwise = pairwise
        self.model_cls = DeepPairwiseImpact if pairwise else DeepImpact

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
        query_terms = self.model_cls.process_query(query=self.queries[qid])

        if self.pairwise:
            for term1, term2 in product(query_terms, query_terms):
                if term1 != term2:
                    query_terms.add(f'{term1}|{term2}')

        scores = self.index.score(query_terms=query_terms)
        self.metrics.add_result(qid=qid, rankings=scores, gold_positives=self.qrels[qid])
        self.metrics.log_metrics()
        return scores
