from pathlib import Path

from src.deep_impact.evaluation import ReRanker

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Evaluate a DeepImpact model by reranking TopK dataset and computing evaluation metrics.")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--top_k_run_file_path", type=Path, required=True)
    parser.add_argument("--queries_path", type=Path, required=True)
    parser.add_argument("--collection_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_processes", type=int, default=4)
    args = parser.parse_args()

    ranker = ReRanker(**vars(args))

    ranker.run()
