from __future__ import annotations

from research.workflows import ProbeWorkflow, TrialWorkflow


def test_workflow_classes_import_and_status_shapes() -> None:
    probe = ProbeWorkflow()
    trial = TrialWorkflow()

    assert probe.status()["phase"] == "created"
    assert probe.status()["completed"] == 0
    assert trial.status()["phase"] == "created"
    assert trial.status()["experiment_id"] is None
