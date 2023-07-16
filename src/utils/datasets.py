from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class MSMarcoTriples():
    def __init__(self, triples_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path]):
        """
        Initialize the MS MARCO Triples Dataset object
        :param triples_path: Path to the triples dataset. Each line is a triple of (qid, pos_id, neg_id)
        :param queries_path: Path to the queries dataset. Each line is a query of (qid, query)
        :param collection_path: Path to the collection dataset. Each line is a passage of (pid, passage)
        """
        self.triples = self._load_triples(triples_path)
        self.queries = self._load_queries(queries_path)
        self.collection = self._load_collection(collection_path)

    @staticmethod
    def _load_triples(path):
        triples = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                qid, pos, neg = map(int, line.strip().split("\t"))
                triples.append((qid, pos, neg))
        return triples

    @staticmethod
    def _load_queries(path):
        queries = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                qid, query = line.strip().split('\t')
                queries[int(qid)] = query
        return queries

    @staticmethod
    def _load_collection(path):
        collection = []
        with open(path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                pid, passage, = line.strip().split('\t')
                assert int(pid) == idx, "Collection is not sorted by id"
                collection.append(passage)
        return collection

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
