from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.datasets import MSMarcoTriples
from src.deep_impact.models import DeepImpact as Model
from src.deep_impact.training import Trainer


def collate_fn(batch, max_length=None):
    encoded_list, masks, labels = [], [], []
    for query, positive_document, negative_document in batch:
        encoded_token, mask = Model.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(1)

        encoded_token, mask = Model.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)
        labels.append(0)

    return encoded_list, torch.stack(masks, dim=0).unsqueeze(-1), torch.tensor(labels, dtype=torch.float)


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
):
    Trainer.ddp_setup()
    dataset = MSMarcoTriples(triples_path, queries_path, collection_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_fn, max_length=max_length),
        sampler=DistributedSampler(dataset),
        drop_last=True,
        num_workers=24,
    )

    model = Model.from_pretrained("bert-base-uncased")
    Model.tokenizer.enable_truncation(max_length=max_length)
    Model.tokenizer.enable_padding(length=max_length)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
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
    Trainer.ddp_cleanup()


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

    args = parser.parse_args()

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
