import argparse
import struct
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.deep_impact.indexing.deep_impact_collection import DeepImpactCollection
from src.utils.defaults import IMPACT_SCORE_FORMAT, DOC_ID_FORMAT, LOC_FORMAT


class InvertedIndexCreator:
    def __init__(self, deep_impact_collection_path: Union[str, Path], output_path: Union[str, Path]):
        self.deep_impact_collection = DeepImpactCollection(Path(deep_impact_collection_path))
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.vocab = dict()

    def _vocab_file(self):
        terms = set()
        for _, item in tqdm(self.deep_impact_collection):
            terms.update(item.keys())

        terms = sorted(terms)
        self.vocab = {term: i for i, term in enumerate(terms)}

        with open(self.output_path / 'vocab.txt', 'w', encoding='utf-8') as f:
            for term in tqdm(self.vocab):
                f.write(f'{term}\n')

    def _inverted_index(self):
        inverted_index = [[] for _ in range(len(self.vocab))]
        for doc_id, item in tqdm(self.deep_impact_collection):
            for term, val in item.items():
                inverted_index[self.vocab[term]].append((doc_id, int(val)))

        start = {}
        end = {}
        with open(self.output_path / 'inverted_index.dat', 'wb') as bf:
            for term_id, docs_and_vals in tqdm(enumerate(inverted_index)):
                for doc_id, val in sorted(docs_and_vals, key=lambda x: x[1], reverse=True):
                    if term_id not in start:
                        start[term_id] = bf.tell()
                    bf.write(struct.pack(DOC_ID_FORMAT, doc_id))  # 4 bytes
                    bf.write(struct.pack(IMPACT_SCORE_FORMAT, val))  # 1 byte
                    end[term_id] = bf.tell()

        with open(self.output_path / 'inverted_index.idx', 'wb') as bf:
            for term_id in tqdm(range(len(self.vocab))):
                bf.write(struct.pack(LOC_FORMAT, start[term_id]))  # 8 bytes
                bf.write(struct.pack(LOC_FORMAT, end[term_id]))  # 8 bytes

    def run(self):
        self._vocab_file()
        self._inverted_index()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--deep_impact_collection_path', type=Path, required=True)
    args.add_argument('-o', '--output_path', type=Path, required=True)
    args = args.parse_args()

    inverted_index_creator = InvertedIndexCreator(
        deep_impact_collection_path=args.deep_impact_collection_path,
        output_path=args.output_path
    )
    inverted_index_creator.run()
