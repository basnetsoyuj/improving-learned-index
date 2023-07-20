import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from tqdm import tqdm


def process(
        input_file_path: Union[str, Path],
        output_file_path: Union[str, Path]
):
    with open(input_file_path) as input_file, open(output_file_path, "w+") as output_file:
        for doc_id, line in tqdm(enumerate(input_file)):
            data = {"id": doc_id, "contents": "", "vector": {}}

            for item in line.strip().split(","):
                term_and_score = item.strip().split(":")
                if len(term_and_score) == 2:
                    term, score = term_and_score
                    data["vector"][term] = float(score)

            json.dump(data, output_file)
            output_file.write('\n')


def main():
    parser = ArgumentParser(description='Convert a DeepImpact collection into an Anserini JsonVectorCollection.')
    parser.add_argument('-i', '--input_file_path', type=Path, required=True)
    parser.add_argument('-o', '--output_file_path', type=Path, required=True)
    args = parser.parse_args()

    process(**vars(args))


if __name__ == "__main__":
    main()
