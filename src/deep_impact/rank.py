from pathlib import Path

from src.deep_impact.evaluation import Ranker
from src.utils.defaults import DATA_DIR

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluate a DeepImpact by ranking qrels docs & computing evaluation metrics.")
    parser.add_argument("--index_path", type=Path, default=DATA_DIR / 'inverted_index')
    parser.add_argument("--qrels_path", type=Path, default=DATA_DIR / 'deep-impact' / 'qrels.dev.small.tsv')
    parser.add_argument("--queries_path", type=Path, default=DATA_DIR / 'deep-impact' / 'queries.dev.tsv')
    parser.add_argument("--output_path", type=Path, default=DATA_DIR / 'reranked.dev.small.tsv')
    parser.add_argument("--pairwise", action='store_true', help="Use pairwise model")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use")
    args = parser.parse_args()

    ranker = Ranker(**vars(args))

    ranker.run()
