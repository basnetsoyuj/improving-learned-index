from .trainer import Trainer


class CrossEncoderTrainer(Trainer):
    def get_output_scores(self, batch):
        input_ids, attention_mask, type_ids = self.get_input_tensors(batch['encoded_list'])
        return self.model(input_ids, attention_mask, type_ids).view(self.batch_size, -1)
