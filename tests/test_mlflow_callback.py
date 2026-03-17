"""Tests for MLFlow callback resume functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from hydra_useful_callbacks.mlflow import MLFlowCallback, MLFlowError


@pytest.fixture
def callback():
    """Create a basic MLFlowCallback with setup mocked out."""
    with patch.object(MLFlowCallback, 'setup'):
        cb = MLFlowCallback(experiment_name='test-exp', tracking_uri='file:///tmp/mlruns')
    return cb


def _make_run_info(run_id='abc123', experiment_id='exp1', status='FINISHED', run_name='my-run'):
    info = SimpleNamespace(
        run_id=run_id, experiment_id=experiment_id, status=status, run_name=run_name
    )
    return SimpleNamespace(info=info)


def _make_experiment(experiment_id='exp1'):
    return SimpleNamespace(experiment_id=experiment_id)


class TestValidateRunForResume:
    """Tests for _validate_run_for_resume."""

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    def test_run_not_found(self, mock_mlflow, callback):
        mock_mlflow.get_run.side_effect = MlflowException('Not found')
        callback.run_id = 'bad-id'
        with pytest.raises(MLFlowError, match='not found'):
            callback._validate_run_for_resume('bad-id')

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    def test_wrong_experiment(self, mock_mlflow, callback):
        mock_mlflow.get_run.return_value = _make_run_info(experiment_id='other-exp')
        mock_mlflow.get_experiment_by_name.return_value = _make_experiment(experiment_id='exp1')
        with pytest.raises(MLFlowError, match='belongs to experiment'):
            callback._validate_run_for_resume('abc123')

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    def test_running_status_warns(self, mock_mlflow, callback, caplog):
        mock_mlflow.get_run.return_value = _make_run_info(status='RUNNING', experiment_id='exp1')
        mock_mlflow.get_experiment_by_name.return_value = _make_experiment(experiment_id='exp1')
        callback._validate_run_for_resume('abc123')
        assert 'RUNNING' in caplog.text

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    def test_finished_status_logs_info(self, mock_mlflow, callback, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            mock_mlflow.get_run.return_value = _make_run_info(status='FINISHED', experiment_id='exp1')
            mock_mlflow.get_experiment_by_name.return_value = _make_experiment(experiment_id='exp1')
            callback._validate_run_for_resume('abc123')
        assert 'Resuming run' in caplog.text

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    def test_experiment_not_yet_created(self, mock_mlflow, callback):
        """When experiment doesn't exist yet, skip experiment-id check."""
        mock_mlflow.get_run.return_value = _make_run_info(experiment_id='exp1')
        mock_mlflow.get_experiment_by_name.return_value = None
        # Should not raise
        callback._validate_run_for_resume('abc123')


def _make_hydra_config(mode, overrides=None, job_num=0, sweep_params=None):
    """Build a minimal mock hydra config."""
    hydra_cfg = SimpleNamespace(
        mode=mode,
        overrides=SimpleNamespace(task=overrides or []),
        sweeper=SimpleNamespace(params=sweep_params),
        job=SimpleNamespace(num=job_num),
        runtime=SimpleNamespace(output_dir='/tmp/out'),
        sweep=SimpleNamespace(dir='/tmp/sweep'),
    )
    config = MagicMock()
    config.hydra = hydra_cfg
    return config


class TestOnMultirunStart:
    """Tests for on_multirun_start with run_id."""

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch.object(MLFlowCallback, 'setup')
    @patch.object(MLFlowCallback, '_validate_run_for_resume')
    def test_resume_parent(self, mock_validate, mock_setup, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='parent-123', nested=True,
            config_file_name=None,
        )
        cb.multiple_jobs = True
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'])
        active = _make_run_info(run_id='parent-123', run_name='original-name')
        mock_mlflow.active_run.return_value = active
        # __wrapped__ bypasses the decorators
        MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config)
        mock_validate.assert_called_once_with('parent-123')
        mock_mlflow.start_run.assert_called_once_with(run_id='parent-123', run_name=None)

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch.object(MLFlowCallback, 'setup')
    @patch.object(MLFlowCallback, '_validate_run_for_resume')
    def test_resume_parent_with_new_name(self, mock_validate, mock_setup, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='parent-123',
            run_name='new-name', nested=True, config_file_name=None,
        )
        cb.multiple_jobs = True
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'])
        active = _make_run_info(run_id='parent-123', run_name='new-name')
        mock_mlflow.active_run.return_value = active
        MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(run_id='parent-123', run_name='new-name')

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch.object(MLFlowCallback, 'setup')
    def test_new_parent_no_run_id(self, mock_setup, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='my-sweep', nested=True,
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'])
        active = _make_run_info(run_id='new-id', run_name='my-sweep')
        mock_mlflow.active_run.return_value = active
        MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(run_name='my-sweep')

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch.object(MLFlowCallback, 'setup')
    def test_run_name_resolved_from_active_run(self, mock_setup, mock_mlflow):
        """When run_name is None and resuming, resolve from the active run."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='parent-123', nested=True,
            config_file_name=None,
        )
        assert cb.run_name is None
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'])
        active = _make_run_info(run_id='parent-123', run_name='original-name')
        mock_mlflow.active_run.return_value = active
        with patch.object(cb, '_validate_run_for_resume'):
            MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config)
        assert cb.run_name == 'original-name'


class TestOnJobStart:
    """Tests for on_job_start with run_id."""

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    @patch.object(MLFlowCallback, '_validate_run_for_resume')
    def test_single_run_resume(self, mock_validate, mock_setup, mock_parse, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='run-456',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        active = _make_run_info(run_id='run-456')
        mock_mlflow.start_run.return_value = active
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        mock_validate.assert_called_once_with('run-456')
        mock_mlflow.start_run.assert_called_once_with(run_id='run-456', run_name=None)

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_single_run_new(self, mock_setup, mock_parse, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='my-run',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        active = _make_run_info(run_id='new-id')
        mock_mlflow.start_run.return_value = active
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(run_name='my-run', nested=False)

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value={'lr': '0.01'})
    @patch.object(MLFlowCallback, 'setup')
    def test_param_conflict_warning(self, mock_setup, mock_parse, mock_mlflow, caplog):
        """When log_params raises on a resumed run, warn instead of crashing."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='run-456',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        active = _make_run_info(run_id='run-456')
        mock_mlflow.start_run.return_value = active
        mock_mlflow.log_params.side_effect = MlflowException('param conflict')
        with patch.object(cb, '_validate_run_for_resume'):
            MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        assert 'param conflict' in caplog.text

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_non_nested_multirun_multiple_jobs_error(self, mock_setup, mock_parse, mock_mlflow):
        """Cannot resume a single run_id across multiple non-nested jobs."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='run-456', nested=False
        )
        cb.multiple_jobs = True
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'])
        with pytest.raises(MLFlowError, match='Cannot resume a single run_id'):
            MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_nested_multirun_children_created(self, mock_setup, mock_parse, mock_mlflow):
        """In nested multirun with run_id, children are new runs (not resumed)."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='parent-123',
            run_name='sweep', nested=True, config_file_name=None,
        )
        cb.multiple_jobs = True
        cb.parent_run_id = 'parent-123'
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'], job_num=0)
        active = _make_run_info(run_id='parent-123')
        mock_mlflow.active_run.return_value = active
        child = _make_run_info(run_id='child-1')
        mock_mlflow.start_run.return_value = child
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        # The start_run for the child should use run_name (not run_id)
        mock_mlflow.start_run.assert_called_with(run_name='sweep-0', nested=True)
