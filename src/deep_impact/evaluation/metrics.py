from collections import defaultdict
from pathlib import Path
from typing import List, Union

from tqdm.auto import tqdm

from src.utils.datasets import QueryRelevanceDataset, TopKDataset, RunFile
from src.utils.logger import Logger

logger = Logger('metrics')


class Metrics:
    def __init__(
            self,
            run_file_path: Union[str, Path],
            qrels_path: Union[str, Path],
            mrr_depths: List[int],
            recall_depths: List[int]
    ):
        self.run_file = RunFile(run_file_path=run_file_path)
        self.qrels = QueryRelevanceDataset(qrels_path=qrels_path)
        self.mrr_sums = {depth: 0 for depth in mrr_depths}
        self.recall_sums = {depth: 0 for depth in recall_depths}

    def evaluate(self):
        """
        Evaluate the metrics.
        :return: None
        """
        relevant_pids_and_ranks = defaultdict(list)
        for qid, pid, rank, _ in self.run_file.read():
            if pid not in self.qrels[qid]:
                continue
            relevant_pids_and_ranks[qid].append(rank)

        for qid, ranks in tqdm(relevant_pids_and_ranks.items()):
            ranks.sort()

            best_rank = ranks[0]
            for depth in self.mrr_sums:
                if best_rank <= depth:
                    self.mrr_sums[depth] += 1.0 / best_rank

            for depth in self.recall_sums:
                positives_upto_depth = len([0 for i in ranks if i <= depth])
                self.recall_sums[depth] += positives_upto_depth / len(self.qrels[qid])

        logger.info(f"\nEvaluated {len(self.qrels)} queries")

        for depth in sorted(self.mrr_sums):
            mrr = round(self.mrr_sums[depth] / len(self.qrels), 3)
            logger.info(f"MRR@{depth} = {mrr}")

        for depth in sorted(self.recall_sums):
            recall = round(self.recall_sums[depth] / len(self.qrels), 3)
            logger.info(f"Recall@{depth} = {recall}")

    @staticmethod
    def evaluate_recall_for_top_k(qrels: QueryRelevanceDataset, top_k: TopKDataset):
        """
        Evaluate the recall at maximum depth for the top k passages for each query.
        :param qrels: The qrels dataset. (Ground truth)
        :param top_k: The top k dataset. (Top K passages for each query)
        :return: The recall for each query.
        """

        assert set(top_k.queries.keys()).issubset(set(qrels.keys())), "TopK file contains queries not in the Qrels file"

        recall = [len(qrels[qid].intersection(set(top_k[qid]))) / len(qrels[qid]) for qid in top_k.keys()]
        recall = round(sum(recall) / len(recall), 3)

        logger.info(f"Recall@{top_k.max_len} = {recall}")
        return recall
