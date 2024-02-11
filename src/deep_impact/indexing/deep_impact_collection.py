from itertools import permutations
from pathlib import Path
from typing import Union, Set


class DeepImpactCollection:
    def __init__(self, index_path: Union[str, Path]):
        self.document_encodings = self._load_document_encodings(index_path)

    @staticmethod
    def _load_document_encodings(index_path: Union[str, Path]):
        document_encodings = []
        with open(index_path, encoding='utf-8') as f:
            for line in f:
                document_encodings.append(line.strip())
        return document_encodings

    def __len__(self):
        return len(self.document_encodings)

    def __getitem__(self, pid):
        string = self.document_encodings[pid]
        if not string.strip():
            return {}
        return {term: float(impact_score) for term, impact_score in (pair.split(': ') for pair in string.split(', '))}

    def score(self, pid, query_terms: Set[str]):
        doc_impacts = self[pid]
        return sum(doc_impacts.get(term, 0) for term in query_terms)

    def __iter__(self):
        for pid in range(len(self)):
            yield pid, self[pid]


class DeepPairwiseImpactCollection(DeepImpactCollection):
    def score(self, pid, query_terms: Set[str]):
        scores_sum = super().score(pid=pid, query_terms=query_terms)

        # add pairwise scores
        doc_impacts = self[pid]
        for term1, term2 in permutations(query_terms, 2):
            scores_sum += doc_impacts.get(f'{term1}|{term2}', 0)

        return scores_sum
