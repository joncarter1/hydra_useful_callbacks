"""MLFlow callback.

Automatically set-up MLFlow logging around Hydra jobs.
"""

__all__ = ('MLFlowCallback',)
import glob
import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Optional
import os
import hydra
import mlflow  # type: ignore
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from .utils import exit_on_error, parse_overrides, rank_zero_only

logger = logging.getLogger(__name__)


class MLFlowError(Exception):
    pass


class MLFlowCallback(Callback):
    """This callback perform MLFlow server set up and tear down for Hydra jobs."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_uri: str,
        artifact_location: str | None = None,
        nested: bool = True,
        resume: bool = False,
        config_file_name: str | None = 'hydra_config.yaml',
        child_run_namer: Optional[str] = None,
    ) -> None:
        """Initialise the callback.

        Args:
            experiment_name: MLFlow experiment to log to.
            run_name: MLFlow run name for single jobs. Parent run name for multi-run jobs.
            tracking_uri: Tracking URI of the MLFlow server to log to.
            artifact_location: Artifact location to log to.
            nested: Whether to nest multiple runs under a parent when in multi-run mode.
            resume: TODO: Resume an existing run. If True, the run_name must be the name of a unique existing run.
            config_file_name: Name for the file used to log the Hydra job config. Or None to skip.
            child_namer: Import path to a Python function for naming MLFlow child runs.
                The imported function should take the same arguments as `default_child_run_namer`.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.artifact_location = artifact_location
        self.parent_run_id = None
        self.child_run_id = None
        self.nested = nested
        self.multiple_jobs = False
        self.resume = resume  # TODO: Implement resume functionality
        self.config_file_name = config_file_name
        if child_run_namer is None:
            self.child_run_namer = default_child_run_namer
        else:
            self.child_run_namer: Callable[[str, DictConfig], str] = hydra.utils.instantiate(  # type: ignore
                {'_target_': child_run_namer, '_partial_': True}
            )
        logger.debug('Created Hydra MLFlow callback.')

    def setup(self) -> None:
        """Set-up MLFlow connection."""
        mlflow.set_tracking_uri(self.tracking_uri)
        if mlflow.get_experiment_by_name(name=self.experiment_name) is None:
            mlflow.create_experiment(name=self.experiment_name, artifact_location=self.artifact_location)
        mlflow.set_experiment(experiment_name=self.experiment_name)

    @exit_on_error
    @rank_zero_only
    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Called in MULTIRUN mode before job/application code starts.

        Start up a parent mlflow run to store sweeps. Runs client-side for remote launchers.
        """
        self.multiple_jobs = _infer_job_count(config) > 1
        self.nested = self.nested and self.multiple_jobs  # Don't nest with only one run
        if not self.nested:
            return
        self.setup()
        if self.resume:
            raise MLFlowError('Resume functionality not yet implemented.')
        else:
            mlflow.start_run(run_name=self.run_name)
        self.parent_run_id = mlflow.active_run().info.run_id
        if self.config_file_name is not None:
            mlflow.log_dict(OmegaConf.to_container(config), self.config_file_name)
            logger.debug('Logged config.')

    @exit_on_error
    @rank_zero_only
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Code executed once at the end of a Hydra sweep."""
        logger.debug('MLFlow multirun callback complete.')

    @exit_on_error
    @rank_zero_only
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Code executed before the start of each individual Hydra job."""
        # Need to re-initialise on server side when launching remotely.
        self.setup()
        # Connect to parent run if using.
        if (config.hydra.mode == RunMode.MULTIRUN) and self.nested:
            if (active_run := mlflow.active_run()) is None:
                logger.debug(f'Connecting to parent run: {self.parent_run_id}')
                mlflow.start_run(run_id=self.parent_run_id)
            elif (run_id := active_run.info.run_id) != self.parent_run_id:
                logger.warning(f'Non-parent {run_id=} already active...')  # Shouldn't be able to get here...

        # Name MLFlow child runs
        if config.hydra.mode == RunMode.MULTIRUN and self.multiple_jobs:
            run_name = self.child_run_namer(parent_run_name=self.run_name, config=config)
        else:
            run_name = self.run_name
        # Only nest if specified and in multi-run mode
        nested = self.nested and config.hydra.mode == RunMode.MULTIRUN
        if self.resume:
            raise MLFlowError('Resume functionality not yet implemented.')
        active_run = mlflow.start_run(run_name=run_name, nested=nested)
        # Set env. var to connect to the run from other modules within the job.
        os.environ['MLFLOW_RUN_ID'] = active_run.info.run_id
        # Fetch the artifact uri root directory
        artifact_uri = mlflow.get_artifact_uri()
        logger.info(f'Starting MLFlow run... Artifacts will be logged to {artifact_uri}.')
        # Log hydra CLI overrides as run parameters.
        override_dict = parse_overrides(config)
        if override_dict is not None:
            mlflow.log_params(override_dict)
        if self.config_file_name is not None:
            mlflow.log_dict(OmegaConf.to_container(config), self.config_file_name)
            logger.debug('Logged job config.')

    @exit_on_error
    @rank_zero_only
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        """Upload log files, including anything from submitit."""
        # Log the output directory logs.
        fps_for_logging = get_files_for_logging(config.hydra.runtime.output_dir)
        # Log .submitit files if present.
        if 'num' in config.hydra.job:  # Check for multirun
            fps_for_logging += get_submitit_files_for_logging(config.hydra.sweep.dir, config.hydra.job.num)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fp in fps_for_logging:
                # Rename to view in MLFlow UI.
                new_fname = os.path.basename(fp).replace('.out', '.stdout.log').replace('.err', '.stderr.log')
                new_fp = os.path.join(tmp_dir, new_fname)
                shutil.copy(fp, new_fp)
                mlflow.log_artifact(new_fp, artifact_path='logs')
        logger.debug(f'Ending MLFlow run: {mlflow.active_run().info.run_id}')
        mlflow.end_run()


def get_files_for_logging(src_dir: str):
    """Find files for logging within a directory."""
    for_logging = []
    for fp in glob.glob(f'{src_dir}/*'):
        if fp.endswith('.out') or fp.endswith('.err') or fp.endswith('.log') or fp.endswith('.pkl'):
            for_logging.append(fp)
    return for_logging


def get_submitit_files_for_logging(sweep_dir: str, job_num: int):
    """Find .submitit files e.g. stdout/stderr associated with job."""
    submitit_dir = os.path.join(sweep_dir, '.submitit')
    if not os.path.exists(submitit_dir):
        logger.debug("Didn't find a .submitit directory. Skipping...")
        return []
    sbatch_fp = glob.glob(f'{submitit_dir}/**/*.sh', recursive=True)[0]
    out = [sbatch_fp]
    slurm_output_dir = glob.glob(f'{submitit_dir}/*_{job_num}')
    # Find the correct submitit subfolder for the given job.
    # Single jobs get logged in the original folder.
    # Otherwise there should be a folder ending in _{job_num} within the .submitit folder.
    if len(slurm_output_dir) == 0:
        if job_num == 0:
            slurm_output_folder = glob.glob(f'{submitit_dir}/*')[0]
            out += get_files_for_logging(slurm_output_folder)
        else:
            logger.warning(f"Didn't a slurm output dir for {job_num=} under {submitit_dir}")
    elif len(slurm_output_dir) > 1:
        logger.warning(f'Found multiple slurm output dirs for {job_num=} under {submitit_dir}')
    else:
        slurm_output_folder = glob.glob(f'{submitit_dir}/*_{job_num}')[0]
        out += get_files_for_logging(slurm_output_folder)
    return out


def _infer_job_count(config: DictConfig) -> int:
    """Infer the number of Hydra jobs from the config.

    Multiplies the number of overrides together.
    """
    job_count = 1
    task_overrides = config.hydra.overrides.task
    cli_keys = []
    if task_overrides is not None:
        for el in task_overrides:
            k, vs = el.split('=')
            cli_keys.append(k)
            sweep_vals = vs.split(',')
            job_count *= len(sweep_vals)
    sweep_config = config.hydra.sweeper.params
    if sweep_config is None:
        return job_count
    for k, vs in sweep_config.items():
        # Ignore values also set on CLI, which takes precedence.
        if k in cli_keys:
            continue
        sweep_vals = vs.split(',')
        job_count *= len(sweep_vals)
    return job_count


def default_child_run_namer(parent_run_name: str, config: DictConfig) -> str:
    """Default function for naming MLFlow child runs from the config."""
    return f'{parent_run_name}-{config.hydra.job.num}'
