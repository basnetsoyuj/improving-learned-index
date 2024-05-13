import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_TYPES
from src.utils.utils import get_unique_query_terms


def merge(collection: Path, collection_type: str, queries: Path, output: Path):
    with open(collection) as f, open(queries) as q, open(output, 'w') as out:
        for line, query_list in tqdm(zip(f, q)):
            doc_id, doc = CollectionParser.parse(line, collection_type)
            query_list = json.loads(query_list)

            assert doc_id == query_list['doc_id'], f"Doc id mismatch: {doc_id} != {query_list['doc_id']}"

            unique_query_terms_str = ' '.join(get_unique_query_terms(query_list['queries'], doc))
            out.write(f'{doc_id}\t{doc} {unique_query_terms_str}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collection with generated queries')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument('--queries_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()
