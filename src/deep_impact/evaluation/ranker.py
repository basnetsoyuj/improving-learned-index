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
            pairwise: bool = False,
    ):
        self.index = InvertedIndex(index_path=index_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.queries = Queries(queries_path=queries_path)
        self.run_file = RunFile(run_file_path=output_path)

        self.pairwise = pairwise
        self.model_cls = DeepPairwiseImpact if pairwise else DeepImpact

    def run(self):
        with tqdm(total=len(self.qrels)) as pbar:
            for qid in self.qrels.keys():
                for rank, (pid, score) in enumerate(self.rank(qid), start=1):
                    self.run_file.write(qid, pid, rank, score)
                pbar.update(1)

    def rank(self, qid: int):
        query_terms = self.model_cls.process_query(query=self.queries[qid])

        if self.pairwise:
            for term1, term2 in product(query_terms, query_terms):
                if term1 != term2:
                    query_terms.add(f'{term1}|{term2}')

        return self.index.score(query_terms=query_terms)
