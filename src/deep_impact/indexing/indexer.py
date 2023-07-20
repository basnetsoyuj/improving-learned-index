import multiprocessing
from pathlib import Path
from typing import Union

import torch
from math import ceil

from src.deep_impact.models.original import DeepImpact as Model
from src.utils.defaults import DEVICE


class Indexer:
    def __init__(
            self,
            model_checkpoint_path: Union[str, Path],
            num_processes: int,
            model_batch_size: int,
    ):
        self.model = Model.load(model_checkpoint_path)
        self.model.to(DEVICE)
        self.model.eval()
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.batch_size = model_batch_size

    @torch.no_grad()
    def index(self, batch, file):
        every_encoded_and_term_to_token_index_map = list(self.pool.map(self.model.process_document, batch))
        every_term_impacts = []

        for batch_idx in range(ceil(len(batch) / self.batch_size)):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_encoded, batch_term_to_token_index_map = zip(*every_encoded_and_term_to_token_index_map[start:end])

            input_ids = torch.tensor([x.ids for x in batch_encoded], dtype=torch.long).to(DEVICE)
            attention_mask = torch.tensor([x.attention_mask for x in batch_encoded], dtype=torch.long).to(DEVICE)
            type_ids = torch.tensor([x.type_ids for x in batch_encoded], dtype=torch.long).to(DEVICE)

            every_term_impacts.extend(self.model.compute_term_impacts(
                documents_term_to_token_index_map=batch_term_to_token_index_map,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=type_ids,
            ))

        lines = []
        for term_impacts in every_term_impacts:
            line = ', '.join([f'{term}: {round(impact, 3)}' for term, impact in term_impacts])
            lines.append(line)

        file.write('\n'.join(lines) + '\n')
        file.flush()
