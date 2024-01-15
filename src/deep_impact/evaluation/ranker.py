from itertools import product
from pathlib import Path
from typing import Union

from tqdm.auto import tqdm

from src.deep_impact.inverted_index import InvertedIndex
from src.deep_impact.models import DeepImpact, DeepPairwiseImpact
from src.utils.datasets import QueryRelevanceDataset, Queries, RunFile


class Ranker:
    def __init__(
            self,
            index_path: Union[str, Path],
            qrels_path: Union[str, Path],
            queries_path: Union[str, Path],
            output_path: Union[str, Path],
            num_workers: int = 4,
            pairwise: bool = False,
    ):
        self.index = InvertedIndex(index_path=index_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.queries = Queries(queries_path=queries_path)
        self.run_file = RunFile(run_file_path=output_path)

        self.num_workers = num_workers
        self.pairwise = pairwise
        self.model_cls = DeepPairwiseImpact if pairwise else DeepImpact

    def run(self):
        for qid in tqdm(self.qrels.keys()):
            scores = self.rank(qid=qid)
            self.run_file.writelines(qid, scores)

    def get_query_terms(self, qid):
        query_terms = self.model_cls.process_query(query=self.queries[qid])

        if self.pairwise:
            for term1, term2 in product(query_terms, query_terms):
                if term1 != term2:
                    query_terms.add(f'{term1}|{term2}')

        return query_terms

    def rank(self, qid):
        return self.index.score(query_terms=self.get_query_terms(qid=qid))
