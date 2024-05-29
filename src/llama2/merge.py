import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_TYPES
from src.utils.utils import merge


def merge_collection_and_expansions(collection_path: Path, collection_type: str, queries_path: Path, output: Path):
    with open(collection_path) as f, open(queries_path) as q, open(output, 'w') as out:
        for line, query_list in tqdm(zip(f, q)):
            doc_id, doc = CollectionParser.parse(line, collection_type)
            query_list = json.loads(query_list)

            assert doc_id == query_list['doc_id'], f"Doc id mismatch: {doc_id} != {query_list['doc_id']}"

            doc = merge(doc, query_list['query'])
            out.write(f'{doc_id}\t{doc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collection with generated queries')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument('--queries_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()

    merge_collection_and_expansions(args.collection_path, args.collection_type, args.queries_path, args.output_path)
