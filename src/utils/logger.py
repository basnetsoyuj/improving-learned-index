import logging
from pathlib import Path


def set_logger(name: str, path: Path, stream: bool = True):
    """
    Define a logger to log output to a file and optionally to the console
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if stream:
        logger.addHandler(logging.StreamHandler())

    return logger
