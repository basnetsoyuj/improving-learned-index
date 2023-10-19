import json
import re
from pathlib import Path
from typing import Union, List
import os
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from src.utils.datasets import Collection
from src.utils.defaults import DEVICE, DATA_DIR


# import torch_xla
# import torch_xla.core.xla_model as xm
# DEVICE = xm.xla_device()

class QueryGenerator:
    def __init__(self, llama_path: Union[str, Path]):
        self.llama_path = Path(llama_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_path)
        self.tokenizer.pad_token = 0  # making it different from the eos token
        self.tokenizer.padding_side = 'left'

        self.model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            load_in_8bit=True,
            device_map=DEVICE,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, documents: List[str], **kwargs):
        assert 'num_return_sequences' in kwargs
        N = kwargs['num_return_sequences']
        inputs = self.prompt_and_tokenize(documents)
        outputs = self.model.generate(**inputs, **kwargs)
        predicted_queries = []
        for d in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
            predicted_queries.append(re.sub(r"\s{2,}", ' ', d.rsplit('\n---\n', 1)[-1]))
        return [predicted_queries[i: i + N] for i in range(0, len(predicted_queries), N)]

    def prompt_and_tokenize(self, documents: List[str]):
        prompts = [f'Predict possible search queries for the following document:\n{document}\n---\n' for document in
                   documents]
        encoded = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True)
        encoded['input_ids'] = encoded['input_ids'].to(DEVICE)
        encoded['attention_mask'] = encoded['attention_mask'].to(DEVICE)
        return encoded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate queries from a trained model.')
    parser.add_argument('--llama_path', type=Path, default=DATA_DIR / 'doc2query-llama-2-7b-merged')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--collection_path', type=Path, default=DATA_DIR / 'collection.tsv')
    parser.add_argument('--output_path', type=Path, default=DATA_DIR / 'queries.tsv')

    args = parser.parse_args()

    generator = QueryGenerator(llama_path=args.llama_path)
    collection = Collection(collection_path=args.collection_path)

    batch = []
    doc_ids = []

    latest_doc_id = 0
    skip = False
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            line = f.readlines()[-1]
        latest_doc_id = json.loads(line)['doc_id']
        skip = True

    for doc_id, document in tqdm(collection):
        if skip:
            if doc_id != latest_doc_id:
                continue
            else:
                skip = False
                continue

        batch.append(document)
        doc_ids.append(doc_id)
        if len(batch) == args.batch_size:
            queries_list = generator.generate(batch, num_return_sequences=80, max_new_tokens=50, do_sample=True,
                                              top_k=50, top_p=0.95)

            with open(args.output_path, 'a', encoding='utf-8') as f:
                for i, queries in enumerate(queries_list):
                    json.dump({'doc_id': doc_ids[i], 'queries': queries}, f)
                    f.write('\n')

            batch = []
            doc_ids = []