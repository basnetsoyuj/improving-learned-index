import torch

from .trainer import Trainer


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def __call__(self, teacher_scores, student_scores):
        """
        Calculates the MSE loss between the teacher and student scores
        :param teacher_scores: Shape (batch_size, 2) with positive and negative scores
        :param student_scores: Shape (batch_size, 2) with positive and negative scores
        :return:
        """
        teacher_margin = -torch.diff(teacher_scores, dim=1).squeeze()
        student_margin = -torch.diff(student_scores, dim=1).squeeze()
        return self.loss(teacher_margin, student_margin)


class DistilTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distil_loss = DistilMarginMSE()

    def evaluate_loss(self, outputs, batch):
        # CrossEntropyLoss
        loss = super().evaluate_loss(outputs, batch)

        # distillation loss
        teacher_scores = batch['scores'].view(self.batch_size, -1).to(self.gpu_id)
        distil_loss = self.distil_loss(teacher_scores, outputs)

        return loss + distil_loss
