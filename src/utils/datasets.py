from pathlib import Path
from typing import Union, Set

from torch.utils.data import Dataset

from src.utils.logger import Logger
from typing import Optional
logger = Logger(__name__)


class Queries:
    """
    Queries dataset.
    :param queries_path: Path to the queries dataset. Each line is a query of (qid, query)
    """

    def __init__(self, queries_path: Union[str, Path]):
        self.queries = self._load_queries(queries_path)

    @staticmethod
    def _load_queries(path):
        queries = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                qid, query = line.strip().split('\t')
                queries[int(qid)] = query
        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, qid):
        return self.queries[qid]


class Collection:
    """
    Collection dataset.
    :param collection_path: Path to the collection dataset. Each line is a passage of (pid, passage)
    """

    def __init__(self, collection_path: Union[str, Path], offset: Optional[int] = None, limit: Optional[int] = None):
        self.collection = self._load_collection(collection_path, offset, limit)

    @staticmethod
    def _load_collection(path, offset: Optional[int] = None, limit: Optional[int] = None):
        if offset is None:
            offset = 0
        if limit is None:
            limit = float('inf')

        collection = {}
        with open(path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < offset:
                    continue
                if idx >= offset + limit:
                    break
                pid, passage, = line.strip().split('\t')
                assert int(pid) == idx, "Collection is not sorted by id"
                collection[int(pid)] = passage
        return collection

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, pid):
        return self.collection[pid]


class MSMarcoTriples(Dataset):
    def __init__(self, triples_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path]):
        """
        Initialize the MS MARCO Triples Dataset object
        :param triples_path: Path to the triples dataset. Each line is a triple of (qid, pos_id, neg_id)
        :param queries_path: Path to the queries dataset. Each line is a query of (qid, query)
        :param collection_path: Path to the collection dataset. Each line is a passage of (pid, passage)
        """
        self.triples = self._load_triples(triples_path)
        self.queries = Queries(queries_path)
        self.collection = Collection(collection_path)

    @staticmethod
    def _load_triples(path):
        triples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                qid, pos, neg = map(int, line.strip().split("\t"))
                triples.append((qid, pos, neg))
        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Get the query, pos and neg items by an index.
        :param idx: The index.
        :return: The query, positive (relevant) doc and negative (non-relevant) doc.
        """
        qid, pos_id, neg_id = self.triples[idx]
        query, positive_doc, negative_doc = self.queries[qid], self.collection[pos_id], self.collection[neg_id]
        return query, positive_doc, negative_doc


class QueryRelevanceDataset:
    def __init__(self, qrels_path: Union[str, Path]):
        """
        Initialize the Qrels Dataset object
        :param qrels_path: Path to the qrels file. Each line: (qid, 0, pid, 1)
        """
        self.qrels = self._load_qrels(qrels_path)

    @staticmethod
    def _load_qrels(path):
        qrels = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, x, pid, y = map(int, line.strip().split('\t'))
                assert x == 0 and y == 1, "Qrels file is not in the expected format"
                qrels.setdefault(qid, set()).add(pid)

        average_positive_per_query = round(sum(len(qrels[qid]) for qid in qrels) / len(qrels), 2)
        logger.info(f"Loaded {len(qrels)} queries with {average_positive_per_query} positive passages/query on average")

        return qrels

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, qid: int) -> Set:
        """
        Get the positive passage ids for a query id.
        :param qid: The query id.
        :return: Set of positive passage ids for the query.
        """
        return self.qrels[qid]

    def keys(self):
        return self.qrels.keys()


class TopKDataset:
    def __init__(self, top_k_path: Union[str, Path]):
        self.queries, self.passages, self.top_k = self._load_topK(top_k_path)

    def _load_topK(self, path):
        queries, passages, top_k = {}, {}, {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, pid, query, passage = line.strip().split('\t')
                qid, pid = int(qid), int(pid)

                assert (qid not in queries) or (queries[qid] == query), "TopK file is not in the expected format"
                queries[qid] = query
                passages[pid] = passage
                top_k.setdefault(qid, []).append(pid)

        assert all(len(top_k[qid]) == len(set(top_k[qid])) for qid in top_k), "TopK file contains duplicates"
        lens = [len(top_k[qid]) for qid in top_k]

        self.min_len = min(lens)
        self.max_len = max(lens)
        self.avg_len = round(sum(lens) / len(top_k), 2)

        logger.info(f"Loaded {len(top_k)} queries:")
        logger.info(f"Top K Passages distribution: min={self.min_len}, max={self.max_len}, avg={self.avg_len}")

        return queries, passages, top_k

    def __len__(self):
        return len(self.top_k)

    def __getitem__(self, qid):
        """
        Get the top k passage ids for a query id.
        :param qid: The query id.
        :return: The top k passage ids for the query.
        """
        return self.top_k[qid]

    def keys(self):
        return self.top_k.keys()
