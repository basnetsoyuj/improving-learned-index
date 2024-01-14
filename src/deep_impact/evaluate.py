from pathlib import Path

from src.deep_impact.evaluation import Metrics
from src.utils.defaults import DATA_DIR

MRR_DEPTHS = [10]
RECALL_DEPTHS = list(range(10, 101, 10))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluate a DeepImpact by ranking qrels docs & computing evaluation metrics.")
    parser.add_argument("--run_file_path", type=Path, default=DATA_DIR / 'run.txt')
    parser.add_argument("--qrels_path", type=Path, default=DATA_DIR / 'deep-impact' / 'qrels.dev.small.tsv')
    args = parser.parse_args()

    evaluator = Metrics(**vars(args))
    evaluator.evaluate()
