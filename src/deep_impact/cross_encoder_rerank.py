from pathlib import Path

from src.deep_impact.evaluation import CrossEncoderReRanker
from src.utils.defaults import DATA_DIR

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Evaluate a Cross Encoder DeepImpact model by reranking TopK dataset.")
    parser.add_argument("--checkpoint_path", type=Path, default=DATA_DIR / 'deep-impact' / 'checkpoint' / 'best.pt')
    parser.add_argument("--top_k_path", type=Path, default=DATA_DIR / 'deep-impact' / 'top1000.dev')
    parser.add_argument("--collection_path", type=Path, default=DATA_DIR / 'collection.tsv')
    parser.add_argument("--output_path", type=Path, default=DATA_DIR / 'run.dev.txt')
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    ranker = CrossEncoderReRanker(**vars(args))

    ranker.run()
