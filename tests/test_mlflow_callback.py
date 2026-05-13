"""Tests for MLFlow callback resume functionality."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from mlflow.exceptions import MlflowException

from hydra_useful_callbacks.mlflow import LOGICAL_KEY_TAG, MLFlowCallback, MLFlowError


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


def _search_results(run_ids=()):
    """Build the list mlflow.search_runs returns with output_format='list'."""
    return [SimpleNamespace(info=SimpleNamespace(run_id=rid)) for rid in run_ids]


def _mock_no_existing_tagged_run(mock_mlflow, experiment_id='exp1'):
    """Configure mocks so the tag-lookup in on_job_start finds nothing."""
    mock_mlflow.get_experiment_by_name.return_value = _make_experiment(experiment_id=experiment_id)
    mock_mlflow.search_runs.return_value = _search_results()


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
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(
            run_name='my-run',
            nested=False,
            tags={LOGICAL_KEY_TAG: f'{cb.host_launch_id}:0'},
        )

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
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        # Child run allocated fresh with logical_key tag scoped by parent_run_id.
        mock_mlflow.start_run.assert_called_with(
            run_name='sweep-0',
            nested=True,
            tags={LOGICAL_KEY_TAG: 'parent-123:0'},
        )


class TestIdempotentJobStart:
    """Tag-based idempotent run allocation across launcher retries (mocked)."""

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_new_run_tagged_with_logical_key(self, mock_setup, mock_parse, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='my-run',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        mock_mlflow.start_run.return_value = _make_run_info(run_id='fresh-id')
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(
            run_name='my-run',
            nested=False,
            tags={LOGICAL_KEY_TAG: f'{cb.host_launch_id}:0'},
        )

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_replay_resumes_tagged_run(self, mock_setup, mock_parse, mock_mlflow):
        """When a tagged run already exists, on_job_start resumes it by id."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='my-run',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        mock_mlflow.get_experiment_by_name.return_value = _make_experiment()
        mock_mlflow.search_runs.return_value = _search_results(['existing-run-id'])
        mock_mlflow.start_run.return_value = _make_run_info(run_id='existing-run-id')
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        mock_mlflow.start_run.assert_called_once_with(run_id='existing-run-id', nested=False)
        # Tags kwarg must NOT be passed when resuming.
        assert 'tags' not in mock_mlflow.start_run.call_args.kwargs

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_replay_uses_same_key_on_second_call(self, mock_setup, mock_parse, mock_mlflow):
        """host_launch_id is stable on the instance, so the logical_key matches across calls."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='my-run',
            config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        mock_mlflow.start_run.return_value = _make_run_info(run_id='id')
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        first_filter = mock_mlflow.search_runs.call_args.kwargs['filter_string']
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        second_filter = mock_mlflow.search_runs.call_args.kwargs['filter_string']
        assert first_filter == second_filter
        assert cb.host_launch_id in first_filter

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_nested_multirun_uses_parent_run_id_for_scope(self, mock_setup, mock_parse, mock_mlflow):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='sweep',
            nested=True, config_file_name=None,
        )
        cb.multiple_jobs = True
        cb.parent_run_id = 'parent-abc'
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'], job_num=2)
        mock_mlflow.active_run.return_value = _make_run_info(run_id='parent-abc')
        mock_mlflow.start_run.return_value = _make_run_info(run_id='child-id')
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        # Filter string must reference the parent-scoped logical key.
        assert "'parent-abc:2'" in mock_mlflow.search_runs.call_args.kwargs['filter_string']
        # And the tag at creation should match.
        assert mock_mlflow.start_run.call_args.kwargs['tags'] == {LOGICAL_KEY_TAG: 'parent-abc:2'}

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    def test_non_nested_multirun_uses_host_launch_id(self, mock_setup, mock_parse, mock_mlflow):
        """Non-nested multirun has no parent_run_id, so the host_launch_id scopes the key."""
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_name='solo',
            nested=False, config_file_name=None,
        )
        cb.multiple_jobs = False  # single-job non-nested multirun
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1'], job_num=0)
        mock_mlflow.start_run.return_value = _make_run_info(run_id='id')
        _mock_no_existing_tagged_run(mock_mlflow)
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        filter_string = mock_mlflow.search_runs.call_args.kwargs['filter_string']
        assert cb.host_launch_id in filter_string

    @patch('hydra_useful_callbacks.mlflow.mlflow')
    @patch('hydra_useful_callbacks.mlflow.parse_overrides', return_value=None)
    @patch.object(MLFlowCallback, 'setup')
    @patch.object(MLFlowCallback, '_validate_run_for_resume')
    def test_user_run_id_takes_precedence_over_lookup(
        self, mock_validate, mock_setup, mock_parse, mock_mlflow
    ):
        from hydra.types import RunMode
        cb = MLFlowCallback(
            experiment_name='test-exp', tracking_uri='uri', run_id='user-x',
            run_name='my-run', config_file_name=None,
        )
        config = _make_hydra_config(mode=RunMode.RUN)
        mock_mlflow.start_run.return_value = _make_run_info(run_id='user-x')
        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        # search_runs should NOT be consulted when user supplied run_id.
        mock_mlflow.search_runs.assert_not_called()
        mock_validate.assert_called_once_with('user-x')
        mock_mlflow.start_run.assert_called_once_with(run_id='user-x', run_name='my-run')


class TestIdempotentJobStartIntegration:
    """End-to-end tests against a real MLflow file store.

    Catches what mocks can't: real filter_string syntax, tag value escaping,
    search_runs round-trip behaviour, idempotent reopen of FINISHED runs.
    """

    @pytest.fixture(autouse=True)
    def _reset_mlflow_state(self):
        while mlflow.active_run() is not None:
            mlflow.end_run()
        os.environ.pop('MLFLOW_RUN_ID', None)
        yield
        while mlflow.active_run() is not None:
            mlflow.end_run()
        os.environ.pop('MLFLOW_RUN_ID', None)

    def _build_callback(self, tmp_path, **kwargs):
        kwargs.setdefault('config_file_name', None)
        return MLFlowCallback(
            experiment_name='test-exp',
            tracking_uri=f'file://{tmp_path}/mlruns',
            **kwargs,
        )

    def test_modal_retry_simulation(self, tmp_path):
        """Two on_job_start calls on the same callback → same MLflow run id."""
        from hydra.types import RunMode
        cb = self._build_callback(tmp_path, run_name='sweep', nested=True)
        config = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'], job_num=0)

        MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config)

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        first_run_id = os.environ['MLFLOW_RUN_ID']
        mlflow.end_run()  # simulate worker exit

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config)
        second_run_id = os.environ['MLFLOW_RUN_ID']

        assert first_run_id == second_run_id, 'Replay must resume the same MLflow run'

    def test_distinct_jobs_get_distinct_runs(self, tmp_path):
        """Different job_nums under the same parent get distinct child runs."""
        from hydra.types import RunMode
        cb = self._build_callback(tmp_path, run_name='sweep', nested=True)
        config_0 = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'], job_num=0)
        config_1 = _make_hydra_config(mode=RunMode.MULTIRUN, overrides=['a=1,2'], job_num=1)

        MLFlowCallback.on_multirun_start.__wrapped__.__wrapped__(cb, config_0)

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config_0)
        run_id_0 = os.environ['MLFLOW_RUN_ID']
        mlflow.end_run()

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb, config_1)
        run_id_1 = os.environ['MLFLOW_RUN_ID']

        assert run_id_0 != run_id_1

    def test_distinct_invocations_get_distinct_runs(self, tmp_path):
        """Fresh callback instances → distinct host_launch_ids → distinct runs."""
        from hydra.types import RunMode
        cb_a = self._build_callback(tmp_path, run_name='solo')
        cb_b = self._build_callback(tmp_path, run_name='solo')
        assert cb_a.host_launch_id != cb_b.host_launch_id

        config = _make_hydra_config(mode=RunMode.RUN)

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb_a, config)
        run_a = os.environ['MLFLOW_RUN_ID']
        mlflow.end_run()

        MLFlowCallback.on_job_start.__wrapped__.__wrapped__(cb_b, config)
        run_b = os.environ['MLFLOW_RUN_ID']

        assert run_a != run_b
