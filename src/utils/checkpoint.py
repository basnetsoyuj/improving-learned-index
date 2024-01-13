from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch

from src.utils.logger import Logger

logger = Logger(Path(__file__).stem, stream=False)


class ModelCheckpoint:
    EXTENSION = 'pt'
    LATEST_SNAPSHOT_SUFFIX = 'latest'

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            checkpoint_dir: Union[str, Path],
            save_every: int,
            filename: Optional[str] = None,
            save_latest_snapshot: bool = True,
            save_best: bool = False,
            batch_size: int = 0,
    ) -> None:
        """
        Initialize the model checkpoint object
        :param model: Torch model
        :param optimizer: Torch optimizer
        :param checkpoint_dir: Path to save the checkpoints
        :param save_every: Save every n steps
        :param filename: Checkpoint prefix
        :param save_latest_snapshot: Save the latest snapshot for torchrun & resuming training
        :param save_best: Save the best model (based on the least metric)
        :param batch_size: Batch size for resuming training
        """
        if filename is None:
            filename = model.__class__.__name__
        self.filename = filename

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.save_every = save_every
        self.save_best = save_best
        self.save_latest_snapshot = save_latest_snapshot
        self.batch_size = batch_size
        self.step = 0

        if self.save_best:
            self.best_metric = float('inf')

    def __call__(self, metric: float = None) -> None:
        self.step += 1

        if self.step % self.save_every == 0:
            self.save(suffix=str(self.step), metric=metric)
            if self.save_latest_snapshot:
                self.save(suffix=self.LATEST_SNAPSHOT_SUFFIX, metric=metric)

        if self.save_best:
            assert metric is not None, "Metric must be provided for save_best=True"
            if metric < self.best_metric:
                self.save(suffix='best', metric=metric)

    def save(self, suffix: str, metric: float = None):
        # to handle DataParallel models
        model = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'batch_size': self.batch_size,
        }
        if metric:
            checkpoint['metric'] = metric

        filename = f'{self.filename}_{suffix}.{self.EXTENSION}'
        torch.save(checkpoint, self.checkpoint_dir / filename)

        logger.info(f'Saved checkpoint to {filename}')

    @classmethod
    def load(
            cls,
            model: torch.nn.Module,
            last_checkpoint_path: Union[str, Path],
            optimizer: Optional[torch.optim.Optimizer] = None,
            save_every: int = 1,
            filename: Optional[str] = None,
            save_latest_snapshot: bool = True,
            save_best: bool = False,
            map_location: Optional[Union[
                Callable[[torch.Tensor, str], torch.Tensor],
                torch.device,
                str,
                Dict[str, str]
            ]] = None,
    ) -> 'ModelCheckpoint':
        """
        Load the model checkpoint
        :param model: Torch model
        :param optimizer: Torch optimizer
        :param last_checkpoint_path: Path to the checkpoint
        :param save_every: Save every n steps
        :param filename: Checkpoint prefix
        :param save_latest_snapshot: Save the latest snapshot for torchrun & resuming training
        :param save_best: Save the best model (based on the least metric)
        :param map_location: Map location for loading the checkpoint
        :return: Model checkpoint object
        """
        last_checkpoint_path = Path(last_checkpoint_path)
        checkpoint = torch.load(last_checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f'Loaded checkpoint for {model.__class__.__name__} from {last_checkpoint_path}')

        obj = cls(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=last_checkpoint_path.parent,
            save_every=save_every,
            filename=filename or last_checkpoint_path.stem.rsplit('_', 1)[0],
            save_latest_snapshot=save_latest_snapshot,
            save_best=save_best,
            batch_size=checkpoint.get('batch_size', 0),
        )
        obj.step = checkpoint.get('step', 0)
        if save_best and 'metric' in checkpoint:
            obj.best_metric = checkpoint['metric']
            obj.save(suffix='best', metric=obj.best_metric)

        return obj
