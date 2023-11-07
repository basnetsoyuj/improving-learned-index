import concurrent.futures
import json
import string
from itertools import product
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.utils.datasets import Collection
from src.utils.defaults import DATA_DIR, DEVICE


@torch.no_grad()
def get_attention_scores(batch_sentences):
    inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)
    return outputs.attentions


def attention_pairs(attention_scores, tokens, index):
    term_ids = []
    terms = []

    for i, token in enumerate(tokens, 1):
        if token.startswith('##'):
            term_ids[-1].append(i)
            terms[-1] += token[2:]
        else:
            term_ids.append([i])
            terms.append(token)

    pairs = {}

    for layer_no, attention_layer in enumerate(attention_scores):
        for i, term1 in zip(term_ids, terms):
            if term1 in ('[CLS]', '[SEP]') or term1 in string.punctuation:
                continue
            for j, term2 in zip(term_ids, terms):
                if term2 in ('[CLS]', '[SEP]') or term2 in string.punctuation:
                    continue

                # if (term1, term2) not in query_trace:
                #     continue

                if i < j:
                    key = (term1, term2, layer_no)

                    # average attention score between all pairs of tokens
                    # val = sum(
                    #     attention_layer[index, :, x, y].mean().item() + attention_layer[index, :, y, x].mean().item()
                    #     for x, y in product(i, j)
                    # ) / (len(i) * len(j) * 2)

                    # max attention
                    val = max(
                        max(attention_layer[index, :, x, y].mean().item(),
                            attention_layer[index, :, y, x].mean().item())
                        for x, y in product(i, j)
                    )

                    if key in pairs:
                        pairs[key] = max(pairs[key], val)
                    elif val > 0.001:
                        pairs[key] = val

    # Sort the pairs by attention score in descending order
    pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    return pairs


def analyze_attention(batch):
    doc_ids, batch_sentences = zip(*batch)
    batch_attention_scores = [x.to('cpu') for x in get_attention_scores(batch_sentences)]
    batch_tokens = [tokenizer.tokenize(sentence) for sentence in batch_sentences]

    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
        results = executor.map(attention_pairs, [batch_attention_scores] * len(batch_tokens), batch_tokens,
                               range(len(batch_tokens)))

    for doc_id, scores in zip(doc_ids, results):
        with open(DATA_DIR / 'attention_scores' / 'scores.tsv', 'a') as f:
            json.dump({'doc_id': doc_id, 'scores': scores}, f)
            f.write('\n')


if __name__ == '__main__':
    punctuations = set(string.punctuation)
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.to(DEVICE)
    model.eval()

    # with open(DATA_DIR / 'pairs_of_terms.pkl', 'rb') as f:
    #     query_trace = pickle.load(f)

    collection = Collection('/hdd1/home/soyuj/expanded_collection.tsv')

    batch_size = 20
    dataloader = collection.batch_iter(batch_size)

    for batch in tqdm(dataloader):
        analyze_attention(batch)
