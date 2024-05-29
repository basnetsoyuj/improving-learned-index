import json
import re
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer

from src.utils.datasets import CollectionParser
from src.utils.defaults import (
    DEVICE,
    COLLECTION_TYPES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_RETURN_SEQUENCES,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P
)


# import torch_xla
# import torch_xla.core.xla_model as xm
# DEVICE = xm.xla_device()

class LLamaQueryGenerator:
    def __init__(self, llama_path: str, max_tokens, peft_path: Optional[str] = None):
        self.llama_path = llama_path
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_path)
        self.tokenizer.pad_token_id = 0  # making it different from the eos token
        self.tokenizer.padding_side = 'left'

        self.model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16,
                # bnb_4bit_use_double_quant=True,
            ),
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
        )

        if peft_path is not None:
            self.peft_config = PeftConfig.from_pretrained(peft_path)
            self.model = PeftModel.from_pretrained(self.model, peft_path)

        self.model.eval()

    @torch.no_grad()
    def generate(self, documents: List[str], **kwargs):
        assert 'num_return_sequences' in kwargs
        n_ret_seq = kwargs['num_return_sequences']
        inputs = self.prompt_and_tokenize(documents)
        outputs = self.model.generate(**inputs, **kwargs)
        predicted_queries = []
        for d in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
            predicted_queries.append(re.sub(r"\s{2,}", ' ', d.rsplit('\n---\n', 1)[-1]))
        return [predicted_queries[i: i + n_ret_seq] for i in range(0, len(predicted_queries), n_ret_seq)]

    @torch.no_grad()
    def prompt_and_tokenize(self, documents: List[str]):
        prompts = [f'Predict possible search queries for the following document:\n{document}' for document in
                   documents]
        encoded = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True,
                                                   max_length=self.max_tokens, truncation=True)

        for input_id in encoded['input_ids']:
            # Check if last three items are not [13, 5634, 13] i.e. \n---\n
            if not torch.equal(input_id[-3:], torch.tensor([13, 5634, 13])):
                # Replace them
                input_id[-3:] = torch.tensor([13, 5634, 13])

        encoded['input_ids'] = encoded['input_ids'].to(DEVICE)
        encoded['attention_mask'] = encoded['attention_mask'].to(DEVICE)
        return encoded


class T5QueryGenerator:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def generate(self, documents: List[str], **kwargs):
        assert 'num_return_sequences' in kwargs
        n_ret_seq = kwargs['num_return_sequences']
        inputs = self.tokenizer.batch_encode_plus(documents, return_tensors='pt', padding=True,
                                                  max_length=self.max_tokens, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
        inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

        outputs = self.model.generate(**inputs, **kwargs)
        predicted_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [predicted_queries[i: i + n_ret_seq] for i in range(0, len(predicted_queries), n_ret_seq)]


def generate_queries_and_save(arguments, query_generator, doc_batch, doc_ids):
    queries_list = query_generator.generate(
        doc_batch,
        num_return_sequences=arguments.num_return_sequences,
        max_new_tokens=arguments.max_new_tokens,
        do_sample=True,
        top_k=arguments.top_k,
        top_p=arguments.top_p
    )

    with open(arguments.output_path, 'a', encoding='utf-8') as out:
        for i, queries in enumerate(queries_list):
            json.dump({'doc_id': doc_ids[i], 'queries': queries}, out)
            out.write('\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate queries from a trained model.')

    # If Peft weights are not merged, pass peft path
    parser.add_argument('--llama_path', type=str, default='./doc2query-llama-2-7b-merged')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument('--output_path', type=Path)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_return_sequences', type=int, default=DEFAULT_NUM_RETURN_SEQUENCES)
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('--peft_path', type=str, default=None)

    args = parser.parse_args()

    generator = LLamaQueryGenerator(llama_path=args.llama_path, max_tokens=args.max_tokens, peft_path=args.peft_path)

    batch = []
    ids = []

    with open(args.collection_path, 'r') as f:
        for line in tqdm(f):
            doc_id, doc = CollectionParser.parse(line, args.collection_type)
            batch.append(doc)
            ids.append(doc_id)

            if len(batch) == args.batch_size:
                generate_queries_and_save(args, generator, batch, ids)
                batch = []
                ids = []

    if batch:
        generate_queries_and_save(args, generator, batch, ids)
