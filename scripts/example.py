"""Example script that uses Hydra for configuration."""

import logging

import hydra
import mlflow
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# Run python example.py --cfg job to print out the composed configuration
# rather than run the application.
@hydra.main(version_base=None, config_path='config', config_name='main')
def main(cfg: DictConfig) -> None:
    logger.info('Starting script.')
    logger.info(f'Val: {cfg.input_val}')
    mlflow.log_param('input_val', cfg.input_val)  # Will log to the run set-up by the callback.
    logger.info('Completed.')


if __name__ == '__main__':
    main()
