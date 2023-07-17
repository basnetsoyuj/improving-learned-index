import os
from pathlib import Path
from typing import Union

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.checkpoint import ModelCheckpoint


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            checkpoint_dir: Union[str, Path],
            batch_size: int,
            save_every: int,
            save_best: bool = True,
            seed: int = 42,
            gradient_accumulation_steps: int = 1,
    ) -> None:
        self.seed = seed
        self.gpu_id = torch.distributed.get_rank()
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.train_data = train_data
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        model_name = self.model.__class__.__name__
        last_checkpoint_path = checkpoint_dir / f'{model_name}_latest.pt'
        if os.path.exists(last_checkpoint_path):
            self.checkpoint_callback = ModelCheckpoint.load(
                model=self.model,
                optimizer=self.optimizer,
                last_checkpoint_path=last_checkpoint_path,
                save_every=save_every,
                save_best=save_best,
            )
        else:
            self.checkpoint_callback = ModelCheckpoint(
                model=self.model,
                optimizer=self.optimizer,
                checkpoint_dir=checkpoint_dir,
                save_every=save_every,
                save_best=save_best,
            )

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def train(self):
        n_ranks = torch.distributed.get_world_size()
        assert self.batch_size % n_ranks == 0, "Batch size must be divisible by the number of GPUs"

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.model.train()

        criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        with tqdm(total=len(self.train_data)) as progress_bar:
            train_loss = 0

            for i, batch in enumerate(self.train_data):
                with torch.cuda.amp.autocast():
                    encoded_list, masks, labels = batch
                    input_ids = torch.tensor([x.ids for x in encoded_list], dtype=torch.long).to(self.gpu_id)
                    attention_mask = torch.tensor([x.attention_mask for x in encoded_list], dtype=torch.long).to(
                        self.gpu_id)
                    type_ids = torch.tensor([x.type_ids for x in encoded_list], dtype=torch.long).to(self.gpu_id)
                    masks = masks.to(self.gpu_id)
                    labels = labels.view(self.batch_size, -1).to(self.gpu_id)

                    document_term_scores = self.model(input_ids, attention_mask, type_ids)
                    outputs = (masks * document_term_scores).sum(dim=1).squeeze(-1).view(self.batch_size, -1)

                    loss = criterion(outputs, labels)
                    loss /= self.gradient_accumulation_steps

                scaler.scale(loss).backward()
                train_loss += loss.item()

                if i % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                if self.gpu_id == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Average Train Loss: {train_loss / (i + 1) * 100:.4f}, "
                        f"Examples Seen: {i * self.batch_size * n_ranks}")
                    self.checkpoint_callback()

    @staticmethod
    def ddp_setup():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(torch.distributed.get_rank())

    @staticmethod
    def ddp_cleanup():
        torch.distributed.destroy_process_group()
