import torch

from .trainer import Trainer


class DistilMarginMSE:
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        self.loss = torch.nn.MSELoss()

    def __call__(self, output, target):
        """
        Calculates the MSE loss between the teacher and student scores
        :param output: Shape (batch_size, 2) with positive and negative predicted scores
        :param target: Shape (batch_size, 2) with positive and negative teacher scores
        :return: MSE loss
        """
        student_margin = -torch.diff(output, dim=1).squeeze()
        teacher_margin = -torch.diff(target, dim=1).squeeze()
        return self.loss(student_margin, teacher_margin)


class DistilKLLoss:
    """Distillation loss from: Distilling Dense Representations for Ranking using Tightly-Coupled Teachers
    link: https://arxiv.org/abs/2010.11386
    """

    def __init__(self):
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def __call__(self, output, target):
        student_scores = torch.log_softmax(output, dim=1)
        teacher_scores = torch.softmax(target, dim=1)
        return self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)


class DistilTrainer(Trainer):
    loss = DistilKLLoss()

    def evaluate_loss(self, outputs, batch):
        teacher_scores = batch['scores'].view(self.batch_size, -1).to(self.gpu_id)
        return self.loss(outputs, teacher_scores)
