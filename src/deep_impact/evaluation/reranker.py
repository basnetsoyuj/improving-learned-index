from pathlib import Path
from typing import Union

from src.deep_impact.indexing.deep_impact_collection import DeepImpactCollection, DeepPairwiseImpactCollection
from src.deep_impact.models import DeepImpact, DeepPairwiseImpact
from src.utils.datasets import QueryRelevanceDataset, TopKDataset
from src.utils.datasets import RunFile


class ReRanker:
    def __init__(
            self,
            index_path: Union[str, Path],
            top_k_path: Union[str, Path],
            qrels_path: Union[str, Path],
            output_path: Union[str, Path],
            pairwise: bool = False,
    ):
        self.index = DeepImpactCollection(index_path=index_path)
        self.top_k = TopKDataset(top_k_path=top_k_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.run_file = RunFile(run_file_path=output_path)

        self.model_cls = DeepPairwiseImpact if pairwise else DeepImpact
        self.collection_cls = DeepPairwiseImpactCollection if pairwise else DeepImpactCollection

    def run(self):
        for qid in self.top_k.keys():
            for rank, (pid, score) in enumerate(self.rerank(qid=qid), start=1):
                self.run_file.write(qid, pid, rank, score)

    def rerank(self, qid: int):
        query = self.top_k.queries[qid]
        query_terms = self.model_cls.process_query(query=query)
        top_k_pids = self.top_k[qid]
        scores = [(pid, self.index.score(pid=pid, query_terms=query_terms)) for pid in top_k_pids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
