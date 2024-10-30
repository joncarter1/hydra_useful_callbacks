"""Timer callback.

Simple job duration logging.
"""

__all__ = ('TimerCallback',)
import logging
import time
from typing import Any

from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TimerCallback(Callback):
    """Simple callback that logs job durations when they complete."""

    def __init__(self) -> None:
        self.start_times: dict[Any, float] = {}

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Store start time."""
        self.start_times[config.hydra.job] = time.perf_counter()

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        """Log elapsed time."""
        completion_time = time.perf_counter() - self.start_times[config.hydra.job]
        if config.hydra.mode == RunMode.MULTIRUN:
            job_name = f'Job {config.hydra.job.num}'
        else:
            job_name = 'Job'
        logger.info(f'{job_name} completed in {completion_time} seconds.')
