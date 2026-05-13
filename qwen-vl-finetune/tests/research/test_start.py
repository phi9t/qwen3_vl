import pytest


def test_build_single_trial_spec_uses_explicit_payload() -> None:
    from research.start import build_single_trial_spec

    spec = build_single_trial_spec(
        profile="b200",
        phase="probe",
        trial_name="smoke",
        env_overrides=["BATCH_SIZE=2", "GRAD_ACCUM_STEPS=4"],
    )

    assert spec == {
        "profile": "b200",
        "phase": "probe",
        "trial": "smoke",
        "env": {
            "BATCH_SIZE": "2",
            "GRAD_ACCUM_STEPS": "4",
        },
    }


def test_build_single_trial_spec_rejects_malformed_env_override() -> None:
    from research.start import build_single_trial_spec

    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        build_single_trial_spec(
            profile="b200",
            phase="probe",
            trial_name="smoke",
            env_overrides=["BATCH_SIZE"],
        )


def test_build_single_trial_spec_requires_trial_name() -> None:
    from research.start import build_single_trial_spec

    with pytest.raises(ValueError, match="trial name is required"):
        build_single_trial_spec(
            profile="b200",
            phase="probe",
            trial_name="",
            env_overrides=[],
        )
