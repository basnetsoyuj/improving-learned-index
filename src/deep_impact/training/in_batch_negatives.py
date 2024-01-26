import torch

from .trainer import Trainer


class InBatchNegativesTrainer(Trainer):
    def get_output_scores(self, batch):
        input_ids, attention_mask, type_ids = self.get_input_tensors(batch['encoded_list'])
        document_term_scores = self.model(input_ids, attention_mask, type_ids).squeeze(-1).view(self.batch_size, 2, -1)

        positive_document_term_scores, negative_document_term_scores = document_term_scores.split(1, dim=1)
        neg_scores_expanded = negative_document_term_scores.transpose(0, 1).expand(self.batch_size, self.batch_size, -1)
        in_batch_term_scores = torch.cat([positive_document_term_scores, neg_scores_expanded], dim=1)
        in_batch_term_scores = in_batch_term_scores.view(self.batch_size * (self.batch_size + 1), -1)

        masks = batch['masks'].to(self.gpu_id)
        return (masks * in_batch_term_scores).sum(dim=1).view(self.batch_size, -1)
