import subprocess
from itertools import compress
from pathlib import Path
from typing import Union

import pyterrier as pt
from pyterrier_doc2query import Doc2QueryStore, QueryScoreStore
from tqdm import tqdm

from src.utils.logger import Logger
from src.utils.utils import get_unique_query_terms

if not pt.started():
    pt.init()


def load_passages(path):
    with open(path, 'r') as f:
        passages = f.readlines()
    return passages


def construct_collection(
        msmarco_passages_path: Union[str, Path],
        queries_repo: str,
        scores_repo: str,
        output_path: Union[str, Path],
        threshold: float,
        unique_terms_only: bool = True
):
    filename = Path(__file__).stem
    logger = Logger(filename, filename, stream=False)

    # for downloading Git-LFS files and not the pointers
    subprocess.run(["git", "lfs", "install", "--skip-repo"])
    queries_store = Doc2QueryStore.from_repo(queries_repo)
    scores_store = QueryScoreStore.from_repo(scores_repo)
    threshold_score = scores_store.percentile(threshold)

    queries_scores_iter = iter(scores_store)

    for line in tqdm(load_passages(msmarco_passages_path)):
        doc_id, passage = line.strip().split('\t')
        item = next(queries_scores_iter)

        assert doc_id == item['docno'], f"Doc id mismatch: {doc_id} != {item['docno']}"

        # filter queries based on threshold
        queries = item['querygen'].split('\n')
        scores_bool = item['querygen_score'] > threshold_score
        queries = list(compress(queries, scores_bool))

        # append queries to passage
        if not unique_terms_only:
            queries_str = ' '.join(queries)
            logger.info(f"{doc_id} | FILTERED QUERIES: {len(queries)}")
        # append only unique terms (injected terms not in passage)
        else:
            unique_terms = get_unique_query_terms(queries, passage)
            queries_str = ' '.join(unique_terms)
            logger.info(f"{doc_id} | FILTERED QUERIES: {len(queries)} | UNIQUE TERMS ADDED: {len(unique_terms)}")

        with open(output_path, 'a') as f:
            f.write(f"{doc_id}\t{passage} [SEP] {queries_str}\n")
