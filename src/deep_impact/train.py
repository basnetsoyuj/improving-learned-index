from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.deep_impact.models import DeepImpact, DeepPairwiseImpact, DeepImpactCrossEncoder
from src.deep_impact.training import Trainer, PairwiseTrainer, CrossEncoderTrainer, DistilTrainer, \
    InBatchNegativesTrainer
from src.deep_impact.training.distil_trainer import DistilMarginMSE, DistilKLLoss
from src.utils.datasets import MSMarcoTriples, DistillationScores, DistilHardNegatives


def collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks = [], []
    for query, positive_document, negative_document in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

        encoded_token, mask = model_cls.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
    }


def cross_encoder_collate_fn(batch):
    encoded_list = []
    for query, positive_document, negative_document in batch:
        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(positive_document, query)
        encoded_list.append(encoded_token)

        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(negative_document, query)
        encoded_list.append(encoded_token)

    return {'encoded_list': encoded_list}


def kl_div_distil_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, scores = [], [], []
    for query, pid_score_list in batch:
        for passage, score in pid_score_list:
            encoded_token, mask = model_cls.process_query_and_document(query, passage, max_length=max_length)
            encoded_list.append(encoded_token)
            masks.append(mask)
            scores.append(score)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


def margin_mse_distil_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, scores = [], [], []
    for query, pos_doc, neg_doc, pos_score, neg_score in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, pos_doc, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        scores.append(pos_score)

        encoded_token, mask = model_cls.process_query_and_document(query, neg_doc, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        scores.append(neg_score)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


def in_batch_negatives_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    queries, positive_documents, negative_documents = zip(*batch)
    queries_terms = [model_cls.process_query(query) for query in queries]
    negatives = [model_cls.process_document(document) for document in negative_documents]

    encoded_list, masks = [], []
    for i, (query_terms, positive_document) in enumerate(zip(queries_terms, positive_documents)):
        encoded_token, term_to_token_index = model_cls.process_document(positive_document)

        encoded_list.append(encoded_token)
        masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

        encoded_list.append(negatives[i][0])
        for _, term_to_token_index in negatives:
            masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0),
    }


def run(
        dataset_path: Union[str, Path],
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
        distil_mse: bool = False,
        distil_kl: bool = False,
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
    if distil_mse:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilMarginMSE()
        collate_function = partial(margin_mse_distil_collate_fn, max_length=max_length)
        dataset_cls = DistilHardNegatives
    elif distil_kl:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(kl_div_distil_collate_fn, max_length=max_length)
        dataset_cls = DistillationScores

    if in_batch_negatives:
        trainer_cls = InBatchNegativesTrainer
        collate_function = partial(in_batch_negatives_collate_fn, max_length=max_length)

    trainer_cls.ddp_setup()
    dataset = dataset_cls(dataset_path, queries_path, collection_path)
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
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the training dataset")
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
    parser.add_argument("--distil_mse", action="store_true", help="Use distillation loss with Mean Squared Error")
    parser.add_argument("--distil_kl", action="store_true", help="Use distillation loss with KL divergence loss")
    parser.add_argument("--in_batch_negatives", action="store_true", help="Use in-batch negatives")
    parser.add_argument("--start_with", type=Path, default=None, help="Start training with this checkpoint")

    # required for distillation loss with Margin MSE
    parser.add_argument("--qrels_path", type=Path, default=None, help="Path to the qrels file")

    args = parser.parse_args()

    assert not (args.distil_mse and args.distil_kl), "Cannot use both distillation losses at the same time"
    assert not (args.distil_kl and args.batch_size != 1), "Can process only 1 example per GPU at a time with KLDiv loss"

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
