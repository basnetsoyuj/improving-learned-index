from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.deep_impact.models import DeepImpact, DeepPairwiseImpact, DeepImpactCrossEncoder
from src.deep_impact.training import Trainer, PairwiseTrainer, CrossEncoderTrainer, DistilTrainer, InBatchNegativesTrainer
from src.utils.datasets import MSMarcoTriples, DistilHardNegatives


def collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, labels = [], [], []
    for query, positive_document, negative_document in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(1)

        encoded_token, mask = model_cls.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(0)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'labels': torch.tensor(labels, dtype=torch.float)
    }


def cross_encoder_collate_fn(batch):
    encoded_list, labels = [], []
    for query, positive_document, negative_document in batch:
        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(positive_document, query)
        encoded_list.append(encoded_token)
        labels.append(1)

        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(negative_document, query)
        encoded_list.append(encoded_token)
        labels.append(0)

    return {'encoded_list': encoded_list, 'labels': torch.tensor(labels, dtype=torch.float)}


def distil_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, labels, scores = [], [], [], []
    for query, positive_document, negative_document, positive_score, negative_score in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(1)
        scores.append(positive_score)

        encoded_token, mask = model_cls.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(0)
        scores.append(negative_score)
    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'labels': torch.tensor(labels, dtype=torch.float),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


def run(
        triples_path: Union[str, Path],
        queries_path: Union[str, Path],
        collection_path: Union[str, Path],
        checkpoint_dir: Union[str, Path],
        max_length: int,
        seed: int,
        batch_size: int,
        lr: float,
        save_every: int,
        save_best: bool,
        gradient_accumulation_steps: int,
        pairwise: bool = False,
        cross_encoder: bool = False,
        distil: bool = False,
        in_batch_negatives: bool = False,
        start_with: Union[str, Path] = None,
):
    # DeepImpact
    model_cls = DeepImpact
    trainer_cls = Trainer
    collate_function = partial(collate_fn, model_cls=DeepImpact, max_length=max_length)
    dataset_cls = MSMarcoTriples

    # Pairwise
    if pairwise:
        model_cls = DeepPairwiseImpact
        trainer_cls = PairwiseTrainer
        collate_function = partial(collate_fn, model_cls=DeepPairwiseImpact, max_length=max_length)

    # CrossEncoder
    elif cross_encoder:
        model_cls = DeepImpactCrossEncoder
        trainer_cls = CrossEncoderTrainer
        collate_function = cross_encoder_collate_fn

    # Use distillation loss
    if distil:
        trainer_cls = DistilTrainer
        collate_function = partial(distil_collate_fn, max_length=max_length)
        dataset_cls = DistilHardNegatives

    if in_batch_negatives:
        trainer_cls = InBatchNegativesTrainer

    trainer_cls.ddp_setup()
    dataset = dataset_cls(triples_path, queries_path, collection_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_function,
        sampler=DistributedSampler(dataset),
        drop_last=True,
        num_workers=0,
    )

    if start_with:
        model = model_cls.load(start_with)
    else:
        model = model_cls.load()
    model_cls.tokenizer.enable_truncation(max_length=max_length, strategy='longest_first')
    model_cls.tokenizer.enable_padding(length=max_length)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        train_data=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        save_every=save_every,
        save_best=save_best,
        seed=seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    trainer.train()
    trainer_cls.ddp_cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Distributed Training of DeepImpact on MS MARCO triples dataset")
    parser.add_argument("--triples_path", type=Path, required=True, help="Path to the triples dataset")
    parser.add_argument("--queries_path", type=Path, required=True, help="Path to the queries dataset")
    parser.add_argument("--collection_path", type=Path, required=True, help="Path to the collection dataset")
    parser.add_argument("--checkpoint_dir", type=Path, required=True, help="Directory to store and load checkpoints")
    parser.add_argument("--max_length", type=int, default=300, help="Max Number of tokens in document")
    parser.add_argument("--seed", type=int, default=42, help="Fix seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=20000, help="Save checkpoint every n steps")
    parser.add_argument("--save_best", action="store_true", help="Save the best model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise training")
    parser.add_argument("--cross_encoder", action="store_true", help="Use cross encoder model")
    parser.add_argument("--distil", action="store_true", help="Use distillation loss")
    parser.add_argument("--in_batch_negatives", action="store_true", help="Use in-batch negatives")
    parser.add_argument("--start_with", type=Path, default=None, help="Start training with this checkpoint")

    args = parser.parse_args()

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
