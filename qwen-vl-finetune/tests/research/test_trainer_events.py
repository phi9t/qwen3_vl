import json
import importlib
import sys
import types
from pathlib import Path


def _load_research_event(monkeypatch):
    class _DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = _DummyModel
    transformers_stub.AutoProcessor = _DummyModel
    transformers_stub.AutoTokenizer = _DummyModel
    transformers_stub.HfArgumentParser = object
    transformers_stub.Qwen2VLForConditionalGeneration = _DummyModel
    transformers_stub.Qwen2_5_VLForConditionalGeneration = _DummyModel
    transformers_stub.Qwen3VLForConditionalGeneration = _DummyModel
    transformers_stub.Qwen3VLMoeForConditionalGeneration = _DummyModel
    transformers_stub.Trainer = object
    transformers_stub.TrainingArguments = object

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        max_memory_allocated=lambda: 0,
        synchronize=lambda: None,
    )
    torch_stub.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    )

    data_processor_stub = types.ModuleType("qwenvl.data.data_processor")
    data_processor_stub.make_supervised_data_module = lambda *args, **kwargs: {}

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "qwenvl.data.data_processor", data_processor_stub)
    monkeypatch.delitem(sys.modules, "qwenvl.train.train_qwen", raising=False)
    return importlib.import_module("qwenvl.train.train_qwen").research_event


def test_research_event_writes_jsonl_when_enabled(monkeypatch, tmp_path: Path) -> None:
    research_event = _load_research_event(monkeypatch)
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("RESEARCH_EVENTS_PATH", str(path))
    monkeypatch.setenv("RESEARCH_TRIAL_ID", "b200/probe/t1")

    research_event("trainer_started", rank=0)

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["schema_version"] == 1
    assert payload["event"] == "trainer_started"
    assert payload["trial_id"] == "b200/probe/t1"
    assert payload["rank"] == 0


def test_research_event_noops_when_disabled(monkeypatch, tmp_path: Path) -> None:
    research_event = _load_research_event(monkeypatch)
    monkeypatch.delenv("RESEARCH_EVENTS_PATH", raising=False)

    research_event("trainer_started")

    assert list(tmp_path.iterdir()) == []
