from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional

from tqdm.auto import tqdm

from src.deep_impact.inverted_index import InvertedIndex
from src.deep_impact.models import DeepImpact, DeepPairwiseImpact
from src.utils.datasets import QueryRelevanceDataset, Queries, RunFile
from src.utils.defaults import COLLECTION_TYPES


def rank(args):
    index, qid, query_terms = args
    return qid, index.score(query_terms=query_terms)


class Ranker:
    def __init__(
            self,
            index_path: Union[str, Path],
            queries_path: Union[str, Path],
            output_path: Union[str, Path],
            num_workers: int = 4,
            qrels_path: Optional[Union[str, Path]] = None,
            pairwise: bool = False,
            dataset_type: Optional[str] = COLLECTION_TYPES[0],
    ):
        self.queries = Queries(queries_path=queries_path, dataset_type=dataset_type)
        self.query_iterator = self.queries.keys()

        # evaluate only the queries in the qrels file
        if qrels_path is not None:
            self.query_iterator = QueryRelevanceDataset(qrels_path=qrels_path).keys()

        self.index = InvertedIndex(index_path=index_path)
        self.run_file = RunFile(run_file_path=output_path)
        self.num_workers = num_workers
        self.pairwise = pairwise
        self.model_cls = DeepPairwiseImpact if pairwise else DeepImpact

    def run(self):
        with Pool(self.num_workers) as p, tqdm(total=len(self.query_iterator)) as pbar:
            for qid, scores in p.imap_unordered(rank, [(self.index, qid, self.get_query_terms(qid)) for qid in
                                                       self.query_iterator]):
                self.run_file.writelines(qid, scores)
                pbar.update(1)

    def get_query_terms(self, qid):
        query_terms = self.model_cls.process_query(query=self.queries[qid])

        if self.pairwise:
            for term1, term2 in product(query_terms, query_terms):
                if term1 != term2:
                    query_terms.add(f'{term1}|{term2}')

        return query_terms
