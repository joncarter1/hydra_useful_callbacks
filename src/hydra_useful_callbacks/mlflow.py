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
from uuid import uuid4

import hydra
import mlflow
from mlflow.exceptions import MlflowException
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from .utils import exit_on_error, parse_overrides, rank_zero_only

logger = logging.getLogger(__name__)

LOGICAL_KEY_TAG = 'hydra_useful_callbacks_logical_key'


class MLFlowError(Exception):
    pass


class MLFlowCallback(Callback):
    """This callback perform MLFlow server set up and tear down for Hydra jobs."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        run_name: str | None = None,
        run_id: str | None = None,
        artifact_location: str | None = None,
        nested: bool = True,
        config_file_name: str | None = 'hydra_config.yaml',
        child_run_namer: Optional[str] = None,
    ) -> None:
        """Initialise the callback.

        Args:
            experiment_name: MLFlow experiment to log to.
            tracking_uri: Tracking URI of the MLFlow server to log to.
            run_name: MLFlow run name for single jobs. Parent run name for multi-run jobs.
                When None and resuming, preserves the existing run's name. When None and
                creating a new run, MLFlow auto-generates one.
            run_id: ID of an existing MLFlow run to resume. When set, the run is reopened
                instead of creating a new one.
            artifact_location: Artifact location to log to.
            nested: Whether to nest multiple runs under a parent when in multi-run mode.
            config_file_name: Name for the file used to log the Hydra job config. Or None to skip.
            child_run_namer: Import path to a Python function for naming MLFlow child runs.
                The imported function should take the same arguments as `default_child_run_namer`.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.run_id = run_id
        self.artifact_location = artifact_location
        self.parent_run_id = None
        self.child_run_id = None
        self.nested = nested
        self.multiple_jobs = False
        self.config_file_name = config_file_name
        # Host-allocated UUID for keying logical jobs in single-run / non-nested
        # multirun modes. Survives the host->worker pickle so launcher replays
        # build the same logical_key and resume the same MLflow run.
        self.host_launch_id: str = uuid4().hex
        if child_run_namer is None:
            self.child_run_namer = default_child_run_namer
        else:
            self.child_run_namer: Callable[[str, DictConfig], str] = hydra.utils.instantiate(  # type: ignore
                {'_target_': child_run_namer, '_partial_': True}
            )
        logger.debug('Created Hydra MLFlow callback.')

    def _validate_run_for_resume(self, run_id: str) -> None:
        """Validate that a run exists and belongs to the configured experiment."""
        try:
            run = mlflow.get_run(run_id)
        except MlflowException:
            raise MLFlowError(f'Cannot resume: run {run_id} not found.')
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None and run.info.experiment_id != experiment.experiment_id:
            raise MLFlowError(
                f'Run {run_id} belongs to experiment {run.info.experiment_id}, '
                f'not "{self.experiment_name}".'
            )
        status = run.info.status
        if status == 'RUNNING':
            logger.warning(f'Run {run_id} has status RUNNING — may be in use by another process.')
        else:
            logger.info(f'Resuming run {run_id} (previous status: {status}).')

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
        if self.run_id is not None:
            self._validate_run_for_resume(self.run_id)
            mlflow.start_run(run_id=self.run_id, run_name=self.run_name)
        else:
            mlflow.start_run(run_name=self.run_name)
        self.parent_run_id = mlflow.active_run().info.run_id
        # Resolve run_name for child naming (e.g. when resuming without a name)
        if self.run_name is None:
            self.run_name = mlflow.active_run().info.run_name
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
        # Detect invalid config: can't resume one run_id across multiple non-nested jobs
        if self.run_id is not None and config.hydra.mode == RunMode.MULTIRUN and not self.nested and self.multiple_jobs:
            raise MLFlowError('Cannot resume a single run_id across multiple non-nested jobs. Use nested=True.')
        if self.run_id is not None and not nested:
            # User-supplied resume (single-run or non-nested multirun with 1 job)
            self._validate_run_for_resume(self.run_id)
            active_run = mlflow.start_run(run_id=self.run_id, run_name=self.run_name)
        else:
            # Idempotent lookup: if a previous attempt of this logical job tagged
            # a run with our logical_key, resume it; otherwise create a fresh run
            # tagged atomically so subsequent replays find it.
            scope_id = self.parent_run_id or self.host_launch_id
            job_num = getattr(config.hydra.job, 'num', 0)
            logical_key = f'{scope_id}:{job_num}'
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            existing = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.{LOGICAL_KEY_TAG} = '{logical_key}'",
                max_results=1,
                output_format='list',  # avoid pandas dep (mlflow-skinny compat)
            )
            if existing:
                logger.info(f'Resuming existing run for {logical_key=}')
                active_run = mlflow.start_run(run_id=existing[0].info.run_id, nested=nested)
            else:
                active_run = mlflow.start_run(
                    run_name=run_name,
                    nested=nested,
                    tags={LOGICAL_KEY_TAG: logical_key},
                )
        # Set env. var to connect to the run from other modules within the job.
        os.environ['MLFLOW_RUN_ID'] = active_run.info.run_id
        # Fetch the artifact uri root directory
        artifact_uri = mlflow.get_artifact_uri()
        logger.info(f'Starting MLFlow run... Artifacts will be logged to {artifact_uri}.')
        # Log hydra CLI overrides as run parameters.
        override_dict = parse_overrides(config)
        if override_dict is not None:
            try:
                mlflow.log_params(override_dict)
            except MlflowException as e:
                logger.warning(f'Could not log overrides (param conflict with existing run): {e}')
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
            # Only split on the first '=' so that override values are
            # allowed to contain '=' characters (e.g. file paths with
            # encoded metadata like 'epoch=04-val_loss=0.52').
            k, vs = el.split('=', maxsplit=1)
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
