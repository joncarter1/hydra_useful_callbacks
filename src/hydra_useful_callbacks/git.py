"""Git callback.

Check for a clean Git history before starting Hydra jobs.
"""

__all__ = ('GitCleanCallback', 'check_git_clean')
import logging
from typing import Any

import git
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from .utils import exit_on_error

logger = logging.getLogger(__name__)


class DirtyGitBranchError(Exception):
    pass


def check_git_clean(override: bool = False) -> bool:
    """Checks a git repo is clean (no unstaged commits)."""
    try:
        repo = git.Repo(search_parent_directories=True)
        git_clean = not repo.is_dirty()
        if git_clean:
            return True
        elif override:
            logger.warning('Git history was dirty but using override=True.')
            return True
        else:
            raise DirtyGitBranchError('Push git commits before running or set override=True in the callback.')
    except Exception as e:
        if override:
            logger.error(f'Checking Git commits failed but override=True - {e}. Continuing...')
            return True
        else:
            raise e


class GitCleanCallback(Callback):
    """Check the Git history is clean before a Hydra job starts."""

    def __init__(self, override: bool = False) -> None:
        self.override = override

    @exit_on_error
    def run(self) -> None:
        check_git_clean(override=self.override)
        logger.debug('Hydra Git callback completed succesffuly.')

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Called in RUN mode before job/application code starts."""
        self.run()

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Called in MULTIRUN mode before job/application code starts."""
        self.run()
