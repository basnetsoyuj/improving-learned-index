import argparse
from src.deep_impact.models.original import DeepImpact
from datasets import load_dataset
from tqdm import tqdm

def create_collection(original_collection_path, output_collection_path):
    with open(original_collection_path, 'r', encoding='utf-8') as f:
        original_collection = [line.strip().split('\t') for line in f]
    
    expansions = load_dataset("pxyu/MSMARCO-TILDE-Top200-CSV300k")['train']
    already_present = 0

    with open(output_collection_path, 'w', encoding='utf-8') as f, tqdm(total=len(original_collection)) as pbar:
        for i, (passage, passage_expansions) in enumerate(zip(original_collection, expansions)):
            assert passage[0] == passage_expansions['pid']
            pre_tokenized_str = DeepImpact.tokenizer.pre_tokenizer.pre_tokenize_str(passage[1])
            terms = {x[0] for x in pre_tokenized_str}
            string_ = ' [SEP]'

            for term in passage_expansions['psg']:
                if term not in terms:
                    string_ += ' ' + term
                else:
                    already_present += 1

            f.write(passage[0] + '\t' + passage[1] + string_ + '\n')

            pbar.update(1)
            pbar.set_description(f"Average duplicate terms per passage: {already_present / (i + 1):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tilde expanded collection.')
    parser.add_argument('--original_collection_path', type=str, help='Path to the original collection')
    parser.add_argument('--output_collection_path', type=str, help='Path to the output collection')
    args = parser.parse_args()
    create_collection(args.original_collection_path, args.output_collection_path)