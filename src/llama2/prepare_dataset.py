import argparse
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.utils.datasets import Queries, Collection, QueryRelevanceDataset
from src.utils.defaults import DEEP_IMPACT_DIR


def process(qrels_path: Union[str, Path], queries_path: Union[str, Path], collection_path: Union[str, Path],
            output_path: Union[str, Path]):
    queries = Queries(queries_path)
    collection = Collection(collection_path)
    dataset = QueryRelevanceDataset(qrels_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in tqdm(dataset.keys()):
            query = queries[qid]
            for doc_id in dataset[qid]:
                f.write(f'{collection[doc_id]}\t{query}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert qrels dataset to literal query-document pairs.')
    parser.add_argument('--qrels_path', type=Path, default=DEEP_IMPACT_DIR / 'qrels.train.tsv')
    parser.add_argument('--queries_path', type=Path, default=DEEP_IMPACT_DIR / 'queries.train.tsv')
    parser.add_argument('--collection_path', type=Path, default=DEEP_IMPACT_DIR / 'collection.tsv')
    parser.add_argument('--output_path', type=Path, default=DEEP_IMPACT_DIR / 'document-query-pairs.train.tsv')
    args = parser.parse_args()

    process(**vars(args))
