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
        return {term: float(impact_score) for term, impact_score in (pair.split(': ') for pair in string.split(', '))}

    def score(self, pid, query_terms: Set[str]):
        return sum(self[pid].get(term, 0) for term in query_terms)
