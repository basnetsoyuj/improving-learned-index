import time
from pathlib import Path
from typing import Union

from src.deep_impact.indexing import Indexer
from src.deep_impact.models import DeepImpact, DeepPairwiseImpact
from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_PATH, DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE, COLLECTION_TYPES
from src.utils.logger import Logger


def run(
        collection_path: Union[str, Path],
        collection_type: str,
        output_file_path: Union[str, Path],
        model_checkpoint_path: Union[str, Path],
        num_processes: int,
        process_batch_size: int,
        model_batch_size: int,
        pairwise: bool = False,
):
    start = time.time()

    logger = Logger(Path(__file__).stem)
    indexer = Indexer(
        model_cls=DeepPairwiseImpact if pairwise else DeepImpact,
        model_checkpoint_path=model_checkpoint_path,
        num_processes=num_processes,
        model_batch_size=model_batch_size,
    )

    with open(collection_path) as f, open(output_file_path, 'w') as out:
        batch = []
        for i, passage in enumerate(f, start=1):
            if i % process_batch_size == 0:
                indexer.index(batch, out)
                logger.info(f'Indexed {i} passages [Rate: {i / (time.time() - start):.2f} passages/s]')
                batch = []

            doc_id, passage = CollectionParser.parse(passage, collection_type)
            batch.append(passage)

        # finish remaining last items
        indexer.index(batch, out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Create a DeepImpact index by computing impacts of all document terms.")
    parser.add_argument("--collection_path", type=Path, default=COLLECTION_PATH,
                        help="Path to the collection dataset")
    parser.add_argument("--collection_type", type=str, default='msmarco', choices=COLLECTION_TYPES,
                        help="Collection dataset type")
    parser.add_argument("--output_file_path", type=Path, default=DATA_DIR / 'collection.index',
                        help="Path to the output file")
    parser.add_argument("--model_checkpoint_path", type=Path, default=CHECKPOINT_DIR / 'DeepImpact_latest.pt',
                        help="Path to the model checkpoint")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use")
    parser.add_argument("--process_batch_size", type=int, default=50 * BATCH_SIZE,
                        help="Batch size for the process pool")
    parser.add_argument("--model_batch_size", type=int, default=BATCH_SIZE, help="Batch size for the model")
    parser.add_argument("--pairwise", action='store_true', help="Use pairwise model")

    args = parser.parse_args()

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
