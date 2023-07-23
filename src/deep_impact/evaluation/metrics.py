from typing import List

from src.utils.datasets import QueryRelevanceDataset, TopKDataset
from src.utils.logger import Logger

logger = Logger('metrics')


class Metrics:
    def __init__(self, mrr_depths: List[int], recall_depths: List[int]):
        self.results = {}
        self.mrr_sums = {depth: 0 for depth in mrr_depths}
        self.recall_sums = {depth: 0 for depth in recall_depths}

    def add_result(self, qid, rankings, gold_positives):
        """
        Add a result to the metrics.
        :param qid: Query ID
        :param rankings: List of (pid, score) tuples
        :param gold_positives: List of positive passage IDs (from Qrels)
        :return: None
        """
        assert qid not in self.results
        assert len(set(gold_positives)) == len(gold_positives)

        self.results[qid] = rankings

        positives = [i for i, (pid, _) in enumerate(rankings) if pid in gold_positives]

        if len(positives) == 0:
            logger.warning(f"Query {qid} has no positive results")
            return

        first_positive_rank = positives[0]

        for depth in self.mrr_sums:
            self.mrr_sums[depth] += (1.0 / (first_positive_rank + 1)) if first_positive_rank < depth else 0.0

        for depth in self.recall_sums:
            positives_upto_depth = len([i for i in positives if i < depth])
            self.recall_sums[depth] += positives_upto_depth / len(gold_positives)

    def log_metrics(self):
        logger.info(f"\nEvaluated {len(self.results)} queries")

        for depth in sorted(self.mrr_sums):
            mrr = round(self.mrr_sums[depth] / len(self.results), 3)
            logger.info(f"MRR@{depth} = {mrr}")

        for depth in sorted(self.recall_sums):
            recall = round(self.recall_sums[depth] / len(self.results), 3)
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

        recall = [len(set(qrels[qid]).intersection(set(top_k[qid]))) / len(qrels[qid]) for qid in top_k.keys()]
        recall = round(sum(recall) / len(recall), 3)

        logger.info(f"Recall@{top_k.max_len} = {recall}")
        return recall
