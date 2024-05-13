import argparse
from pathlib import Path
from typing import Union, Optional

from tqdm import tqdm

from src.utils.defaults import IMPACT_SCORE_QUANTIZATION_BITS, DATA_DIR
from src.utils.logger import Logger

logger = Logger('quantize')


def quantize(value: float, scale: float):
    return int(value * scale)


def find_max_value(input_file_path: Union[str, Path]):
    max_val = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for doc_id, line in tqdm(enumerate(f)):
            for t in line.strip().split(', '):
                _, score = t.strip().split(': ')
                max_val = max(max_val, float(score))
    return max_val


def quantize_file(
        input_file_path: Union[str, Path],
        output_file_path: Union[str, Path],
        max_val: Optional[float] = None):
    if max_val is None:
        max_val = find_max_value(input_file_path)
        logger.info(f'Found max value: {max_val}')
    else:
        logger.info(f'Using given max value: {max_val}')

    scale = ((1 << IMPACT_SCORE_QUANTIZATION_BITS) - 1) / max_val

    with open(input_file_path, 'r', encoding='utf-8') as f, open(output_file_path, 'w', encoding='utf-8') as out:
        for doc_id, line in tqdm(enumerate(f)):
            data = []
            for t in line.strip().split(', '):
                term, score = t.strip().split(': ')
                val = quantize(float(score), scale)
                if val > 0:
                    data.append(f'{term}: {val}')
            out.write(', '.join(data) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize a DeepImpact collection.')
    parser.add_argument('-i', '--input_file_path', type=Path, default=DATA_DIR / 'collection.index')
    parser.add_argument('-o', '--output_file_path', type=Path, default=DATA_DIR / 'collection.index.quantized')
    parser.add_argument('-m', '--max_val', type=float, default=None)

    args = parser.parse_args()

    quantize_file(**vars(args))
