import argparse
import struct
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.deep_impact.indexing.deep_impact_collection import DeepImpactCollection


class InvertedIndexCreator:
    def __init__(self, deep_impact_collection_path: Union[str, Path], output_path: Union[str, Path]):
        self.deep_impact_collection = DeepImpactCollection(deep_impact_collection_path)
        self.output_path = output_path
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
        inverted_index = []
        for i, item in tqdm(self.deep_impact_collection):
            for term, val in item.items():
                inverted_index.append((term, i, val))

        inverted_index.sort(key=lambda x: (x[0], -x[2]))

        with open(self.output_path / 'inverted_index.txt', 'w', encoding='utf-8') as f:
            for term, i, val in tqdm(inverted_index):
                f.write(f'{term}\t{i}\t{val}\n')

        start = {}
        end = {}
        with open(self.output_path / 'inverted_index.dat', 'wb') as bf:
            for term, i, val in tqdm(inverted_index):
                term_id = self.vocab[term]
                if term_id not in start:
                    start[term_id] = bf.tell()
                bf.write(struct.pack('I', i))  # 4 bytes
                bf.write(struct.pack('B', int(val)))  # 1 byte
                end[term_id] = bf.tell()

        with open('inverted_index.idx', 'wb') as bf:
            for term_id in tqdm(range(len(self.vocab))):
                bf.write(struct.pack('I', term_id))  # 4 bytes
                bf.write(struct.pack('Q', start[term_id]))  # 8 bytes
                bf.write(struct.pack('Q', end[term_id]))  # 8 bytes

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
