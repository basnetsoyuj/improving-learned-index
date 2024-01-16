import argparse
import gzip
import json
import random
from pathlib import Path

from tqdm import tqdm

from src.utils.defaults import DEEP_IMPACT_DIR


def run(negatives_path: Path, output_path: Path):
    list_ = []

    with gzip.open(negatives_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            qid = data['qid']
            pos_ids = data['pos']

            neg = set()
            for method, neg_ids in data['neg'].items():
                neg.update(neg_ids)

            list_.extend([(qid, pid, nid) for pid in pos_ids for nid in neg])

    # shuffle list
    random.shuffle(list_)

    print(f'Writing to {output_path}')
    with open(output_path, 'w', encoding='utf-8') as f:
        for qid, pid, nid in tqdm(list_):
            f.write(f'{qid}\t{pid}\t{nid}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construct new dataset to train DeepImpact using hard negatives and distillation.')
    parser.add_argument('--negatives_path', type=Path, default=DEEP_IMPACT_DIR / 'msmarco-hard-negatives.jsonl.gz')
    parser.add_argument('--output_path', type=Path, default=DEEP_IMPACT_DIR / 'hard_neg.tsv')

    run(**vars(parser.parse_args()))
