"""Unsupported chat methods must skip through the current runner, without engine dependencies."""
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
for directory in ("hf", "ortgenai"):
    sys.path.insert(0, str(ROOT / "adapters" / directory))
import ir_conformance as conf  # noqa: E402


@pytest.mark.parametrize("module,classname", [
    ("hf_adapter", "HFAdapter"),
    ("ortgenai_adapter", "ORTGenAIAdapter"),
])
def test_external_adapter_chat_cases_skip_without_aborting(module, classname):
    cls = getattr(importlib.import_module(module), classname)
    # These methods need no model or backend. Avoid importing/initialising heavy engines.
    adapter = cls.__new__(cls)
    counts = conf.run_suite(adapter, only="chat/")
    assert counts["SKIP"] == 86
    assert sum(counts.values()) == counts["SKIP"]


@pytest.mark.parametrize("module,classname", [
    ("hf_adapter", "HFAdapter"),
    ("ortgenai_adapter", "ORTGenAIAdapter"),
])
def test_external_adapter_accepts_explicit_tool_definitions(module, classname):
    cls = getattr(importlib.import_module(module), classname)
    adapter = cls.__new__(cls)
    with pytest.raises(NotImplementedError, match="template"):
        adapter.render_chat({}, [], True, None, [{"name": "lookup"}])
    with pytest.raises(NotImplementedError, match="template"):
        adapter.render_chat({}, [], tools=[{"name": "lookup"}])
