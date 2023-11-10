import torch

from .trainer import Trainer


class PairwiseTrainer(Trainer):
    def get_output_scores(self, input_ids, attention_mask, type_ids, masks):
        pairwise_indices = []
        for document_mask in masks.squeeze(-1):
            pairwise_indices.append([])
            nonzero_indices = document_mask.nonzero().squeeze(-1)
            combinations = torch.combinations(nonzero_indices, r=2)
            pairwise_indices[-1].extend(combinations.tolist())
            pairwise_indices[-1].extend(combinations.flip(dims=(1,)).tolist())

        document_term_scores, pairwise_term_scores = self.model(
            input_ids,
            attention_mask,
            type_ids,
            pairwise_indices
        )

        outputs = (masks * document_term_scores).sum(dim=1).view(self.batch_size, -1)

        pairwise_outputs = torch.stack(
            [scores_per_doc.sum(dim=0) for scores_per_doc in pairwise_term_scores],
            dim=0
        ).view(self.batch_size, -1)

        return outputs + pairwise_outputs
