from pathlib import Path

import torch

DEVICE = torch.device('cuda:0')

LOG_DIR = Path(__file__).parent.parent.parent / 'logs'

DATA_DIR = Path('/hdd1/home/soyuj/')
COLLECTION_PATH = DATA_DIR / 'expanded-collection.tsv'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'

BATCH_SIZE = 32
