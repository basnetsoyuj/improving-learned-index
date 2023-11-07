import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualize attention scores.')
parser.add_argument('--input_path', type=str, default='/hdd1/home/soyuj/attention_scores/test.tsv')
parser.add_argument('--doc_id', type=int, default=0)
args = parser.parse_args()

data = {}

with open(args.input_path, 'r') as file:
    for i, line in enumerate(file):
        if i <= args.doc_id:
            json_content = json.loads(line)
            data[json_content['doc_id']] = [i[1] for i in json_content['scores']]
        else:
            break

sns.set_style('whitegrid')

for (doc_id, scores) in tqdm(data.items()):
    mean_val = np.mean(scores)
    median_val = np.median(scores)
    max_val = np.max(scores)
    min_val = np.min(scores)

    stats_text = f'Mean: {mean_val:.8f}, Median: {median_val:.8f}\nMax: {max_val:.8f}, Min: {min_val:.8f}'

    plt.figure(figsize=(10, 7))

    sns.histplot(scores, bins=200, color='skyblue', kde=True)

    plt.title(f'Doc ID: {doc_id}, len:{len(scores)}\n{stats_text}', fontsize=15)
    plt.xlabel("Scores", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)

    plt.savefig(f'/hdd1/home/soyuj/visualization/{doc_id}_hist.png')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(scores)
    plt.title(f'Doc ID: {doc_id}, len:{len(scores)}\n{stats_text}', fontsize=15)
    plt.xlabel("Term Pairs", fontsize=15)
    plt.ylabel("Scores", fontsize=15)
    plt.savefig(f'/hdd1/home/soyuj/visualization/{doc_id}_plot.png')
    plt.close()
