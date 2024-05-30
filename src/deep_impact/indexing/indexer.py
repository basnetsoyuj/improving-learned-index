import multiprocessing
from math import ceil
from pathlib import Path
from typing import Union

import torch
from torch.nn import DataParallel

from src.utils.defaults import DEVICE


class Indexer:
    def __init__(
            self,
            model_cls: torch.nn.Module,
            model_checkpoint_path: str,
            num_processes: int,
            model_batch_size: int,
    ):
        self.model_cls = model_cls
        self.device = torch.device("cuda")
        self.model = model_cls.load(model_checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)

        self.pool = multiprocessing.Pool(processes=num_processes)
        self.batch_size = model_batch_size

    @torch.no_grad()
    def index(self, batch, file):
        every_encoded_and_term_to_token_index_map = list(self.pool.map(self.model_cls.process_document, batch))
        every_term_impacts = []

        for batch_idx in range(ceil(len(batch) / self.batch_size)):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_encoded, batch_term_to_token_index_map = zip(*every_encoded_and_term_to_token_index_map[start:end])

            input_ids = torch.tensor([x.ids for x in batch_encoded], dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([x.attention_mask for x in batch_encoded], dtype=torch.long).to(DEVICE)
            type_ids = torch.tensor([x.type_ids for x in batch_encoded], dtype=torch.long).to(DEVICE)

            # ------------------ DeepImpact ------------------
            outputs = self.model(input_ids, attention_mask, type_ids)
            # ---------------------------------------------------

            # ------------------ DeepPairwiseImpact ------------------
            # pairwise_indices = [
            #     list(combinations(sorted(map_.values()), r=2))
            #     for map_ in batch_term_to_token_index_map
            # ]
            # outputs = self.model(input_ids, attention_mask, type_ids, pairwise_indices)
            # ---------------------------------------------------

            every_term_impacts.extend(self.model_cls.compute_term_impacts(
                documents_term_to_token_index_map=batch_term_to_token_index_map,
                outputs=outputs,
            ))

        lines = []
        for term_impacts in every_term_impacts:
            line = ', '.join([f'{term}: {round(impact, 3)}' for term, impact in term_impacts])
            lines.append(line)

        file.write('\n'.join(lines) + '\n')
        file.flush()
