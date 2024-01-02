import multiprocessing
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.deep_impact.models import DeepPairwiseImpact as Model


class Indexer:
    def __init__(
            self,
            model_checkpoint_path: Union[str, Path],
            num_processes: int,
            model_batch_size: int,
    ):
        self.device = torch.device("cuda")  # torch.device("cuda:1,3")
        self.model = Model.load(model_checkpoint_path)
        self.model.eval()
        self.model = nn.DataParallel(self.model)  # nn.DataParallel(self.model, device_ids=[1,3])
        self.model.to(self.device)
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.batch_size = model_batch_size

    @torch.no_grad()
    def index(self, batch, file):
        every_encoded_and_term_to_token_index_map = list(self.pool.map(self.model.module.process_document, batch))
        every_term_impacts = []

        for batch_idx in range(ceil(len(batch) / self.batch_size)):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_encoded, batch_term_to_token_index_map = zip(*every_encoded_and_term_to_token_index_map[start:end])

            input_ids = torch.tensor([x.ids for x in batch_encoded], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([x.attention_mask for x in batch_encoded], dtype=torch.long).to(self.device)
            type_ids = torch.tensor([x.type_ids for x in batch_encoded], dtype=torch.long).to(self.device)

            # ------------------ DeepImpact ------------------
            # outputs = self.model(input_ids, attention_mask. type_ids)
            # ---------------------------------------------------

            # ------------------ DeepPairwiseImpact ------------------
            pairwise_indices = [
                list(combinations(sorted(map_.values()), r=2))
                for map_ in batch_term_to_token_index_map
            ]
            outputs = self.model(input_ids, attention_mask, type_ids, pairwise_indices)
            # ---------------------------------------------------

            every_term_impacts.extend(self.model.module.compute_term_impacts(
                documents_term_to_token_index_map=batch_term_to_token_index_map,
                outputs=outputs,
            ))

        lines = []
        for term_impacts in every_term_impacts:
            line = ', '.join([f'{term}: {round(impact, 3)}' for term, impact in term_impacts])
            lines.append(line)

        file.write('\n'.join(lines) + '\n')
        file.flush()
