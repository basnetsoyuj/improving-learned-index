from .trainer import Trainer


class CrossEncoderTrainer(Trainer):
    def get_output_scores(self, batch):
        encoded_list, labels = batch
        input_ids, attention_mask, type_ids = self.get_input_tensors(encoded_list)
        output = self.model.module.get_bert_output(input_ids, attention_mask, type_ids)

        # get CLS token output
        output = output.last_hidden_state[:, 0, :]
        document_term_scores = self.model.module.get_impact_scores(output)

        return document_term_scores.view(self.batch_size, -1)
