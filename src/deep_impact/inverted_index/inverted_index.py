import heapq
import multiprocessing as mp
import struct
from functools import lru_cache
from pathlib import Path
from typing import Union

from src.utils.defaults import (
    INVERTED_INDEX_VOCAB,
    INVERTED_INDEX_INDEX,
    INVERTED_INDEX_DATA,
    LOC_BLOCK_BYTES,
    LOC_BLOCK_FORMAT,
    DOC_SCORE_BLOCK_BYTES,
    DOC_SCORE_BLOCK_FORMAT,
)


class InvertedIndex:
    def __init__(self, index_path: Union[str, Path]):
        self.index_path = Path(index_path)
        self.vocab = self._load_vocab()

    def _load_vocab(self):
        vocab = dict()
        with open(self.index_path / INVERTED_INDEX_VOCAB, encoding='utf-8') as f:
            for i, line in enumerate(f):
                vocab[line.strip()] = i
        return vocab

    def term_location(self, term):
        term_id = self.vocab.get(term, None)
        if term_id is None:
            return None, None, None
        with open(self.index_path / INVERTED_INDEX_INDEX, 'rb') as bf:
            bf.seek(term_id * LOC_BLOCK_BYTES)
            start, end = struct.unpack(LOC_BLOCK_FORMAT, bf.read(LOC_BLOCK_BYTES))
        return term_id, start, end

    # @lru_cache(maxsize=10000)
    def term_docs(self, term):
        term_id, start, end = self.term_location(term)
        if term_id is None:
            return []
        with open(self.index_path / INVERTED_INDEX_DATA, 'rb') as bf:
            bf.seek(start)
            docs = []
            while bf.tell() < end:
                doc_id, value = struct.unpack(DOC_SCORE_BLOCK_FORMAT, bf.read(DOC_SCORE_BLOCK_BYTES))
                if value == 0:
                    break
                docs.append((doc_id, value))
        return docs

    def score(self, query_terms, top_k=1000):
        scores = {}
        term_results = map(self.term_docs, query_terms)
        for term, docs in zip(query_terms, term_results):
            for doc_id, score in docs:
                scores[doc_id] = scores.get(doc_id, 0) + score

        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
