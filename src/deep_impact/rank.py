from pathlib import Path

from src.deep_impact.evaluation import Ranker
from src.utils.defaults import DATA_DIR, COLLECTION_TYPES

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluate a DeepImpact by ranking qrels docs & computing evaluation metrics.")
    parser.add_argument("--index_path", type=Path, default=DATA_DIR / 'inverted_index')
    parser.add_argument("--queries_path", type=Path, default=DATA_DIR / 'deep-impact' / 'queries.dev.tsv')
    parser.add_argument("--output_path", type=Path, default=DATA_DIR / 'run.tsv')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use")
    parser.add_argument("--qrels_path", type=Path, default=None)
    parser.add_argument("--dataset_type", type=str, default=COLLECTION_TYPES[0], choices=COLLECTION_TYPES)
    parser.add_argument("--pairwise", action='store_true', help="Use pairwise model")

    args = parser.parse_args()

    ranker = Ranker(**vars(args))

    ranker.run()
