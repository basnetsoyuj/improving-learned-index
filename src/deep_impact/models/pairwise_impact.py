from itertools import combinations
from typing import List, Tuple, Dict

import torch
import torch.nn as nn

from .original import DeepImpact


class DeepPairwiseImpact(DeepImpact):
    def __init__(self, config):
        super(DeepPairwiseImpact, self).__init__(config)
        self.pairwise_impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + 1, 1),
            nn.ReLU()
        )
        self.init_weights()

    # noinspection PyMethodOverriding
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            pairwise_indices: List[List[List[int]]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :param pairwise_indices: List of pairwise indices
        :return: Batch of impact scores and pairwise impact scores
        """

        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids, output_attentions=True)
        single_impact_scores = self._get_term_impact_scores(bert_output.last_hidden_state)
        pairwise_impact_scores, pairwise_attentions = self.get_pairwise_impact_scores(
            bert_output.last_hidden_state,
            bert_output.attentions,
            pairwise_indices
        )
        return single_impact_scores, pairwise_impact_scores, pairwise_attentions

    def get_pairwise_impact_scores(
            self,
            last_hidden_state: torch.Tensor,
            attentions: Tuple[torch.Tensor, ...],
            pairwise_indices: List[List[List[int]]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        :param last_hidden_state: Last hidden state from BERT
        :param attentions: Tuple of attention matrices
        :param pairwise_indices: List of pairwise indices
        :return: Pairwise impact scores
        """
        pairwise_attention_with_hidden_states = []
        pairwise_attentions = []

        for i, pairwise_indices_per_doc in enumerate(pairwise_indices):
            attention_scores = []
            for pairs in pairwise_indices_per_doc:
                max_attention = max(
                    max(
                        attentions[layer][i, :, pairs[0], pairs[1]].mean(),
                        attentions[layer][i, :, pairs[1], pairs[0]].mean()
                    )
                    for layer in range(len(attentions))
                )
                max_attention = max_attention.detach().view(1)
                attention_scores.append(max_attention)

                pairwise_attention_with_hidden_states.append(
                    torch.cat(
                        (
                            max_attention,
                            last_hidden_state[i, pairs].flatten()
                        )
                    )
                )

            if len(attention_scores) == 0:
                pairwise_attentions.append(last_hidden_state.new([]))
            else:
                pairwise_attentions.append(torch.cat(attention_scores, dim=0).view(-1, 1))

        pairwise_attention_with_hidden_states = torch.stack(pairwise_attention_with_hidden_states, dim=0)
        output = self.pairwise_impact_score_encoder(pairwise_attention_with_hidden_states)

        pairwise_scores = []
        start = 0
        for doc_pairs in pairwise_indices:
            end = start + len(doc_pairs)
            pairwise_scores.append(output[start:end])
            start = end
        return pairwise_scores, pairwise_attentions

    @staticmethod
    def compute_term_impacts(
            documents_term_to_token_index_map: List[Dict[str, int]],
            outputs: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
    ) -> List[List[Tuple[str, float]]]:
        """
        Computes the impact scores of each term in each document
        :param documents_term_to_token_index_map: List of dictionaries mapping each unique term to its first token index
        :param outputs: Tuple of batch of impact scores and batch of pairwise impact scores and pairwise indices
        :return: Batch of lists of tuples of document terms and their impact scores
        """
        impact_scores, pairwise_impact_scores, _ = outputs
        impact_scores = impact_scores.squeeze(-1).cpu().numpy()
        pairwise_impact_scores = [x.squeeze(-1).cpu().numpy() for x in pairwise_impact_scores]

        term_impacts = []
        for i, map_ in enumerate(documents_term_to_token_index_map):
            term_impacts.append([
                (term, impact_scores[i][token_index])
                for term, token_index in map_.items()
            ])

        for i, map_ in enumerate(documents_term_to_token_index_map):
            for j, ((term1, _), (term2, _)) in enumerate(combinations(sorted(map_.items(), key=lambda x: x[1]), r=2)):
                score = pairwise_impact_scores[i][j]
                if round(score, 3):
                    term_impacts[i].append(
                        (f'{term1}|{term2}', pairwise_impact_scores[i][j])
                    )

        for i in range(len(term_impacts)):
            term_impacts[i] = sorted(term_impacts[i], key=lambda x: x[1], reverse=True)
        return term_impacts
