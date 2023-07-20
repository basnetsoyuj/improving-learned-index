from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import tokenizers
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from src.utils.checkpoint import ModelCheckpoint


class DeepImpact(BertPreTrainedModel):
    max_length = 512
    tokenizer = tokenizers.Tokenizer.from_pretrained('bert-base-uncased')

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
            max_length = DeepImpact.max_length

        query = DeepImpact.tokenizer.normalizer.normalize_str(query)
        query_terms = {x[0] for x in DeepImpact.tokenizer.pre_tokenizer.pre_tokenize_str(query)}

        encoded, document_terms, term_index_to_token_index = DeepImpact._encode_document_and_map_term_to_tokens(
            document)

        token_indices_of_matching_terms = []
        seen = set()
        for i, term in enumerate(document_terms):
            if (term in query_terms) and (term not in seen) and (i in term_index_to_token_index):
                token_indices_of_matching_terms.append(term_index_to_token_index[i])
                seen.add(term)

        mask = np.zeros(max_length, dtype=bool)
        mask[token_indices_of_matching_terms] = True

        return encoded, torch.from_numpy(mask)

    @staticmethod
    def process_document(document: str) -> Tuple[tokenizers.Encoding, List[Tuple[str, int]]]:
        """
        Encodes the document and maps each term to its corresponding first token index.
        :param document: Document string
        :return: Tuple: Encoded document, List of tuples of document terms and their first token index
        """
        encoded, document_terms, map_ = DeepImpact._encode_document_and_map_term_to_tokens(document)

        seen = set()
        term_to_token_index = []

        # filter out duplicate terms and terms whose tokens overflow the max length
        for i, term in enumerate(document_terms):
            if term not in seen and i in map_:
                seen.add(term)
                term_to_token_index.append((term, map_[i]))
        return encoded, term_to_token_index

    @staticmethod
    def _encode_document_and_map_term_to_tokens(document: str) -> (tokenizers.Encoding, List[str], Dict[int, int]):
        """
        Encodes the document and maps each term index to its corresponding first token index.
        :param document: Document string
        :return: Tuple: Encoded document, List of document terms, Dict mapping term index to first token index
        """

        document = DeepImpact.tokenizer.normalizer.normalize_str(document)
        document_terms = [x[0] for x in DeepImpact.tokenizer.pre_tokenizer.pre_tokenize_str(document)]

        encoded = DeepImpact.tokenizer.encode(document_terms, is_pretokenized=True)

        term_index_to_token_index = {}
        counter = 0
        # encoded[1:] because the first token is always the CLS token
        for i, token in enumerate(encoded.tokens[1:], start=1):
            if token.startswith("##"):
                continue
            term_index_to_token_index[counter] = i
            counter += 1

        return encoded, document_terms, term_index_to_token_index

    @staticmethod
    def load(checkpoint_path: Optional[Union[str, Path]] = None):
        model = DeepImpact.from_pretrained("bert-base-uncased")
        if checkpoint_path is not None:
            ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
        DeepImpact.tokenizer.enable_truncation(max_length=DeepImpact.max_length)
        DeepImpact.tokenizer.enable_padding(length=DeepImpact.max_length)
        return model

    @torch.no_grad()
    def compute_term_impacts(
            self,
            documents_terms: List[List[Tuple[str, int]]],
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> List[List[Tuple[str, float]]]:
        """
        Computes the impact scores of each term in each document
        :param documents_terms: Batch of lists of tuples of document terms and their first token index
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :return: Batch of lists of tuples of document terms and their impact scores
        """
        impact_scores = self.forward(input_ids, attention_mask, token_type_ids).squeeze(-1)
        impact_scores = impact_scores.cpu().numpy()

        term_impacts = []
        for i, document_terms in enumerate(documents_terms):
            term_impacts.append([(term, impact_scores[i][token_index]) for term, token_index in document_terms])

        return term_impacts
