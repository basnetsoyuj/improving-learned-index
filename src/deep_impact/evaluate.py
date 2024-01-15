from pathlib import Path

from src.deep_impact.evaluation import Metrics
from src.utils.defaults import DATA_DIR

MRR_DEPTHS = [10]
RECALL_DEPTHS = [10, 20, 50] + list(range(100, 1001, 100))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluate a DeepImpact by ranking qrels docs & computing evaluation metrics.")
    parser.add_argument("--run_file_path", type=Path, default=DATA_DIR / 'run.txt')
    parser.add_argument("--qrels_path", type=Path, default=DATA_DIR / 'deep-impact' / 'qrels.dev.small.tsv')
    args = parser.parse_args()

    evaluator = Metrics(**vars(args), mrr_depths=MRR_DEPTHS, recall_depths=RECALL_DEPTHS)
    evaluator.evaluate()
