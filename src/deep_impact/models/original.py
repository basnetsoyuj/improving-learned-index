import string
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Set

import tokenizers
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from src.utils.checkpoint import ModelCheckpoint


class DeepImpact(BertPreTrainedModel):
    max_length = 512
    tokenizer = tokenizers.Tokenizer.from_pretrained('bert-base-uncased')
    punctuation = set(string.punctuation)

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
    ) -> torch.Tensor:
        """
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :return: Batch of impact scores
        """
        bert_output = self.get_bert_output(input_ids, attention_mask, token_type_ids)
        return self.get_impact_scores(bert_output.last_hidden_state)

    def get_bert_output(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :param output_attentions: Whether to output attentions
        :return: Batch of BERT outputs
        """
        return self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

    def get_impact_scores(
            self,
            last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param last_hidden_state: Last hidden state from BERT
        :return: Impact scores
        """
        return self.impact_score_encoder(last_hidden_state)

    @classmethod
    def process_query_and_document(cls, query: str, document: str) -> (torch.Tensor, torch.Tensor):
        """
        Process query and document to feed to the model
        :param query: Query string
        :param document: Document string
        :return: Tuple: Document Tokens, Grouped Tokens for query-document intersection terms
        """
        query_terms = cls.process_query(query)
        encoded, term_to_token_indices = cls.process_document(document)

        return encoded, cls.get_query_document_token_mask(query_terms, term_to_token_indices)

    @classmethod
    def get_query_document_token_mask(cls, query_terms: Set[str],
                                      term_to_token_indices: Dict[str, List[int]]) -> List[List[int]]:
        return [v for k, v in term_to_token_indices.items() if k in query_terms]

    @classmethod
    def process_query(cls, query: str) -> Set[str]:
        query = cls.tokenizer.normalizer.normalize_str(query)
        return set(filter(lambda x: x not in cls.punctuation,
                          map(lambda x: x[0], cls.tokenizer.pre_tokenizer.pre_tokenize_str(query))))

    @classmethod
    def process_document(cls, document: str) -> Tuple[tokenizers.Encoding, Dict[str, List[int]]]:
        """
        Encodes the document and maps each unique term (non-punctuation) to its corresponding first token's index.
        :param document: Document string
        :return: Tuple: Encoded document, Dict mapping unique non-punctuation document terms to token indices
        """

        document = cls.tokenizer.normalizer.normalize_str(document)
        document_terms = [x[0] for x in cls.tokenizer.pre_tokenizer.pre_tokenize_str(document)]

        encoded = cls.tokenizer.encode(document_terms, is_pretokenized=True)

        term_index_to_token_index = {}
        counter = 0
        # encoded[1:] because the first token is always the CLS token
        for i, token in enumerate(encoded.tokens[1:], start=1):
            if token.startswith("##"):
                continue
            term_index_to_token_index[counter] = i
            counter += 1

        filtered_term_to_token_indices = {}

        # filter out duplicate terms, punctuations, and terms whose tokens overflow the max length
        for i, term in enumerate(document_terms):
            if term not in cls.punctuation and i in term_index_to_token_index:
                if term not in filtered_term_to_token_indices:
                    filtered_term_to_token_indices[term] = [term_index_to_token_index[i]]
                else:
                    filtered_term_to_token_indices[term].append(term_index_to_token_index[i])
        return encoded, filtered_term_to_token_indices

    @classmethod
    def load(cls, checkpoint_path: Optional[Union[str, Path]] = None):
        model = cls.from_pretrained("Luyu/co-condenser-marco")
        if checkpoint_path is not None:
            ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
        cls.tokenizer.enable_truncation(max_length=cls.max_length, strategy='longest_first')
        cls.tokenizer.enable_padding(length=cls.max_length)
        return model

    @staticmethod
    def compute_term_impacts(
            documents_term_to_token_indices_map: List[Dict[str, List[int]]],
            outputs: torch.Tensor,
    ) -> List[List[Tuple[str, float]]]:
        """
        Computes the impact scores of each term in each document
        :param documents_term_to_token_indices_map: List of dict mapping each unique term to list of token indices
        :param outputs: Batch of model outputs
        :return: Batch of lists of tuples of document terms and their impact scores
        """
        impact_scores = outputs.squeeze(-1).cpu().numpy()

        term_impacts = []
        for i, term_to_token_index_map in enumerate(documents_term_to_token_indices_map):
            term_impacts.append([
                (term, impact_scores[i][token_indices].max())
                for term, token_indices in term_to_token_index_map.items()
            ])

        return term_impacts
