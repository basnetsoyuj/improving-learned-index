import logging
from pathlib import Path

DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / 'logs'


def Logger(name: str, filename: str, stream: bool = True, log_dir: Path = DEFAULT_LOG_DIR) -> logging.Logger:
    """
    Define a logger to log output to a file and optionally to the console
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f'{filename}.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if stream:
        logger.addHandler(logging.StreamHandler())

    return logger
