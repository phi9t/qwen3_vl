def test_temporal_modules_import() -> None:
    from research.activities import (
        load_profile_activity,
        plan_probe_trials_activity,
        run_trial_activity,
    )
    from research.workflows import ProfiledExperimentWorkflow, SingleTrialWorkflow

    assert load_profile_activity is not None
    assert plan_probe_trials_activity is not None
    assert run_trial_activity is not None
    assert ProfiledExperimentWorkflow is not None
    assert SingleTrialWorkflow is not None


def test_workflow_status_shape_includes_latest_analysis() -> None:
    from research.workflows import ProfiledExperimentWorkflow

    workflow = ProfiledExperimentWorkflow()
    status = workflow.status()

    assert "latest_metrics" in status
    assert "latest_analysis" in status
    assert status["latest_analysis"] is None


def test_single_trial_workflow_status_shape() -> None:
    from research.workflows import SingleTrialWorkflow

    workflow = SingleTrialWorkflow()
    status = workflow.status()

    assert status == {
        "phase": "created",
        "active_trial": "",
        "completed_trials": 0,
        "latest_metrics": None,
        "latest_analysis": None,
    }
