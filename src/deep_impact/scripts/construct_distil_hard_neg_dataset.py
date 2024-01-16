import argparse
import gzip
import pickle
import random
from pathlib import Path

from tqdm import tqdm

from src.utils.datasets import QueryRelevanceDataset
from src.utils.defaults import DEEP_IMPACT_DIR


def run(qrels_path: Path, scores_path: Path, output_path: Path):
    qrels = QueryRelevanceDataset(qrels_path=qrels_path)

    with gzip.open(scores_path, 'rb') as f:
        scores = pickle.load(f)

    list_ = []
    positive_scores = dict()

    for qid in tqdm(qrels.keys()):
        positive_scores[qid] = {pid: scores[qid].pop(pid) for pid in qrels[qid]}
        list_.extend([(qid, pid, nid) for pid in qrels[qid] for nid in scores[qid].keys()])

    print(f'Number of triples with teacher scores: {len(list_)}')

    # shuffle list
    random.shuffle(list_)

    print(f'Writing to {output_path}')
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid, pid, nid in tqdm(list_):
            f.write(f'{qid}\t{pid}\t{nid}\t{positive_scores[qid][pid]}\t{scores[qid][nid]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construct new dataset to train DeepImpact using hard negatives and distillation.')

    parser.add_argument('--qrels_path', type=Path, default=DEEP_IMPACT_DIR / 'qrels.train.tsv')
    parser.add_argument('--scores_path', type=Path,
                        default=DEEP_IMPACT_DIR / 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    parser.add_argument('--output_path', type=Path, default=DEEP_IMPACT_DIR / 'distil_hard_neg.tsv')

    run(**vars(parser.parse_args()))
