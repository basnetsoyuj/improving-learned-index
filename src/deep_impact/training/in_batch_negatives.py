import torch

from .trainer import Trainer


class InBatchNegativesTrainer(Trainer):
    def evaluate_loss(self, outputs, batch):
        batch_size = outputs.size(0)

        pos_scores, neg_scores = outputs.split(1, dim=-1)
        neg_scores_expanded = neg_scores.transpose(0, 1).expand(batch_size, batch_size)
        in_batch_scores = torch.cat([pos_scores, neg_scores_expanded], dim=-1)

        labels = torch.zeros(batch_size, dtype=torch.long).to(self.gpu_id)
        return self.criterion(in_batch_scores, labels)
