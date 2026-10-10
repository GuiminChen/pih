import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("v41_batch_config", Path(__file__).parents[2] / "tools/configure_v41_batch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("sequences", [1, 2, 4, 32])
def test_upgrades_configuration_preserving_rank_and_budget_identity(sequences):
    original = {"schema": "pih.deepseek-v41.supervisor.v1", "ranks": [{"device_budget": 123}], "model": {"plugin_lock_sha256": "abc"}}
    result = module.configure(original, sequences, 4096, 64)
    assert result["schema"] == "pih.deepseek-v41.supervisor.v2"
    assert result["batching"]["maximum_batch_sequences"] == sequences
    assert result["ranks"] == original["ranks"] and result["model"] == original["model"]
    assert "batching" not in original


@pytest.mark.parametrize("limits", [(0,4096,64), (33,4096,64), (4,3,64), (4,4097,64), (4,4096,0), (4,4096,4097), (True,4096,64)])
def test_rejects_unreserved_limits(limits):
    with pytest.raises(ValueError):
        module.configure({"schema": "pih.deepseek-v41.supervisor.v1"}, *limits)
