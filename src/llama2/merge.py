import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_TYPES
from src.utils.utils import get_unique_query_terms


def merge(collection_path: Path, collection_type: str, queries_path: Path, output: Path):
    with open(collection_path) as f, open(queries_path) as q, open(output, 'w') as out:
        for line, query_list in tqdm(zip(f, q)):
            doc_id, doc = CollectionParser.parse(line, collection_type)
            query_list = json.loads(query_list)

            assert doc_id == query_list['doc_id'], f"Doc id mismatch: {doc_id} != {query_list['doc_id']}"

            doc = doc.replace('\n', ' ')
            unique_query_terms_str = ' '.join(get_unique_query_terms(query_list['queries'], doc))

            doc = re.sub(r"\s{2,}", ' ', f'{doc} {unique_query_terms_str}')
            out.write(f'{doc_id}\t{doc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collection with generated queries')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument('--queries_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()

    merge(args.collection_path, args.collection_type, args.queries_path, args.output_path)
