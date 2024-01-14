import tokenizers
import torch

from .original import DeepImpact


class DeepImpactCrossEncoder(DeepImpact):
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
        return self.get_impact_scores(bert_output.last_hidden_state[:, 0, :])

    @classmethod
    def process_cross_encoder_document_and_query(
            cls,
            document: str,
            query: str,
    ) -> tokenizers.Encoding:
        """
        Encodes the document and query for cross-encoder models
        :param document: Document string
        :param query: Query string
        :return: Encoded document and query
        """
        return cls.tokenizer.encode(f'{document} [SEP] {query}')
