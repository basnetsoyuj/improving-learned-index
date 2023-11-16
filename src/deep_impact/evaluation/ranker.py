from pathlib import Path
from typing import Union

from src.deep_impact.evaluation.metrics import Metrics
from src.deep_impact.indexing.deep_impact_collection import DeepPairwiseImpactCollection as DeepImpactCollection
from src.deep_impact.models import DeepPairwiseImpact as Model
from src.utils.datasets import QueryRelevanceDataset, TopKDataset


class Ranker:
    def __init__(
            self,
            index_path: Union[str, Path],
            top_k_path: Union[str, Path],
            qrels_path: Union[str, Path],
            output_path: Union[str, Path],
    ):
        self.index = DeepImpactCollection(index_path=index_path)
        self.top_k = TopKDataset(top_k_path=top_k_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.metrics = Metrics(mrr_depths=[10], recall_depths=[50, 200, 500])
        self.output_path = output_path

        self.metrics.evaluate_recall_for_top_k(qrels=self.qrels, top_k=self.top_k)

    def run(self):
        for qid in self.top_k.keys():
            rankings = self.rerank(qid=qid)
            with open(self.output_path, 'a') as f:
                for pid, score in rankings:
                    f.write(f'{qid}\t{pid}\t{score}\n')

            # print(f'Query {self.top_k.queries[qid]}')
            # print('Gold positives:')
            # for pid in self.qrels[qid]:
            #     print(pid)
            # print('Rankings:')
            # for pid, score in rankings[:10]:
            #     print(f'{score}\t{self.top_k.passages[pid]}')

    def rerank(self, qid: int):
        query = self.top_k.queries[qid]
        query_terms = Model.process_query(query=query)
        top_k_pids = self.top_k[qid]
        scores = [(pid, self.index.score(pid=pid, query_terms=query_terms)) for pid in top_k_pids]
        scores.sort(key=lambda x: x[1], reverse=True)

        self.metrics.add_result(qid=qid, rankings=scores, gold_positives=self.qrels[qid])
        self.metrics.log_metrics()
        return scores
