from pathlib import Path

import torch

DEVICE = torch.device('cuda:0')

LOG_DIR = Path(__file__).parent.parent.parent / 'logs'

DATA_DIR = Path('/hdd1/home/soyuj/')
DEEP_IMPACT_DIR = DATA_DIR / 'deep-impact'

COLLECTION_PATH = DATA_DIR / 'expanded-collection.tsv'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'

BATCH_SIZE = 32

QUANTIZATION_BITS = 8

LLAMA_DIR = DATA_DIR / 'llama2'
LLAMA_HUGGINGFACE_CHECKPOINT = LLAMA_DIR / 'models_hf' / '7B'
