from pathlib import Path

from src.deep_impact.evaluation.ranker import Ranker
from src.utils.defaults import DATA_DIR

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Evaluate a DeepImpact model by reranking TopK dataset and computing evaluation metrics.")
    parser.add_argument("--index_path", type=Path, default=DATA_DIR / 'collection.index')
    parser.add_argument("--top_k_path", type=Path, default=DATA_DIR / 'deep-impact' / 'top1000.dev')
    parser.add_argument("--qrels_path", type=Path, default=DATA_DIR / 'deep-impact' / 'qrels.dev.small.tsv')
    parser.add_argument("--output_path", type=Path, default=DATA_DIR / 'reranked.dev.small.tsv')
    args = parser.parse_args()

    ranker = Ranker(**vars(args))

    ranker.run()
