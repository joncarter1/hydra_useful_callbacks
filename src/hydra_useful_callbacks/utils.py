"""Callback utility functions."""

import ast
import logging
import traceback
from functools import wraps
import os

logger = logging.getLogger(__name__)


def is_rank_zero():
    """Check if the current process is rank zero.

    This is a non-exhaustive check that assumes a PyTorch environment, which may be on a Slurm Cluster.
    """
    for key in (
        'RANK',
        'LOCAL_RANK',
        'SLURM_PROCID',
    ):
        rank = os.environ.get(key)
        if rank is not None and int(rank) == 0:
            return True

    return True


def rank_zero_only(method):
    """Decorator to perform commands only on rank zero process."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if is_rank_zero():
            return method(self, *args, **kwargs)
        else:
            logger.debug('Not running on non-rank-zero process.')
            return

    return wrapper


def exit_on_error(method):
    """Exit Hydra callback methods on errors rathering than powering through (default behaviour).

    Logs the stack trace.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            stack_trace = ''.join(traceback.format_exception(type(e), e, tb=None))
            logger.error(stack_trace)
            exit()

    return wrapper


def parse_overrides(cfg):
    """Parse the Hydra overrides."""
    try:
        overrides = {}
        for override in cfg.hydra.overrides.task:
            key, value = override.split('=', maxsplit=1)
            # Fudge to remove problematic characters from Hydra syntax for MLFlow logging.
            key = key.replace('@', 'AT').replace('/', '_').replace('+', '').replace('~', '-')
            overrides[key] = try_cast(value)
        return overrides
    except Exception as e:
        logger.error(f'Failed to parse overrides from config - {e}.')
        logger.error(cfg.hydra.overrides.task)
        return None


def try_cast(val):
    """Return a valid Python type from string if possible."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    except Exception as e:
        logger.error(f'Failed to handle {val}')
        raise e
