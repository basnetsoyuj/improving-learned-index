from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import BertPreTrainedModel, BertModel


class DeepImpact(BertPreTrainedModel):
    tokenizer = Tokenizer.from_pretrained('bert-base-uncased')

    def __init__(self, config):
        super(DeepImpact, self).__init__(config)
        self.bert = BertModel(config)
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )
        self.init_weights()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        return self.impact_score_encoder(x)

    @staticmethod
    def process_query_and_document(query: str, document: str, max_length: Optional[int] = None) -> \
            (torch.Tensor, torch.Tensor):
        """
        Process query and document to feed to the model
        :param query: Query string
        :param document: Document string
        :param max_length: Max number of tokens to process
        :return: Tuple: Document Tokens, Mask with 1s corresponding to first tokens of document terms in the query
        """
        if max_length is None:
            max_length = 512

        query = DeepImpact.tokenizer.normalizer.normalize_str(query)
        query_terms = {x[0] for x in DeepImpact.tokenizer.pre_tokenizer.pre_tokenize_str(query)}

        document = DeepImpact.tokenizer.normalizer.normalize_str(document)
        document_terms = [x[0] for x in DeepImpact.tokenizer.pre_tokenizer.pre_tokenize_str(document)][:max_length]

        encoded = DeepImpact.tokenizer.encode(" ".join(document_terms))

        first_token_index_of_term = {}
        counter = 0
        # encoded[1:] because the first token is always the CLS token
        for i, token in enumerate(encoded.tokens[1:], start=1):
            if token.startswith("##"):
                continue
            first_token_index_of_term[counter] = i
            counter += 1

        token_indices_of_matching_terms = []
        seen = set()
        for i, term in enumerate(document_terms):
            if (term in query_terms) and (term not in seen) and (i in first_token_index_of_term):
                token_indices_of_matching_terms.append(first_token_index_of_term[i])
                seen.add(term)

        mask = np.zeros(max_length, dtype=bool)
        mask[token_indices_of_matching_terms] = True

        return encoded, torch.from_numpy(mask)
