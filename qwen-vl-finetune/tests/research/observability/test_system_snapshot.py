import subprocess

from research.observability.system_snapshot import collect_system_snapshot


def test_collect_system_snapshot_parses_gpu_query(monkeypatch) -> None:
    def fake_check_output(cmd, text=True, stderr=None):
        joined = " ".join(cmd)
        if "--query-gpu" in joined:
            return "0, NVIDIA B200, 183359, 10, 0\n1, NVIDIA B200, 183359, 20, 5\n"
        if cmd[:2] == ["nvidia-smi", "--query"]:
            return ""
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    payload = collect_system_snapshot(
        env={"BATCH_SIZE": "2", "HF_TOKEN": "secret"},
        include_process_topology=False,
    )

    assert payload["schema_version"] == 1
    assert payload["gpus"][0]["name"] == "NVIDIA B200"
    assert payload["gpus"][1]["memory_used_mb"] == 20
    assert payload["env"]["HF_TOKEN"] == "***REDACTED***"
    assert "process_topology" not in payload


def test_collect_system_snapshot_can_include_process_topology(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "")

    payload = collect_system_snapshot(
        env={},
        include_process_topology=True,
        process_topology=[{"pid": 1}],
    )

    assert payload["process_topology"] == [{"pid": 1}]
