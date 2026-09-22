"""Build ownership guards, not link or cross-plugin capability qualification."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
OWNERS = (
    ("execution-default", "pih_execution_default_impl", 8),
    ("storage-verified-artifact", "pih_storage_verified_artifact_impl", 6),
    ("memory-host-spill", "pih_memory_host_spill_impl", 1),
    ("platform-linux", "pih_platform_linux_impl", 26),
)


@pytest.mark.parametrize("owner,target,count", OWNERS)
def test_implementation_is_declared_only_by_owner(owner, target, count):
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f"include(plugins/{owner}/implementation.cmake)" in root_cmake
    definition = re.compile(r"add_library\(\s*" + re.escape(target) + r"\s+([^)]*)\)")
    assert not definition.search(root_cmake)
    owner_dir = ROOT / "plugins" / owner
    source = (owner_dir / "implementation.cmake").read_text(encoding="utf-8")
    matches = definition.findall(source)
    assert len(matches) == 1
    names = re.findall(r'"\$\{CMAKE_CURRENT_LIST_DIR\}/([^"\n]+\.cpp)"', matches[0])
    assert len(names) == len(set(names)) == count
    for name in names:
        path = (owner_dir / name).resolve()
        assert path.is_relative_to(owner_dir.resolve())
        assert path.is_file()
    assert "${CMAKE_CURRENT_SOURCE_DIR}" not in source


def test_output_credit_archive_has_same_owner_as_execution():
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (ROOT / "plugins/execution-default/implementation.cmake").read_text(encoding="utf-8")
    assert "add_library(pih_output_burst_credits" not in root_cmake
    assert 'add_library(pih_output_burst_credits STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/output_burst_credit_pool.cpp")' in source
    assert root_cmake.index("include(plugins/execution-default/implementation.cmake)") < root_cmake.index("add_subdirectory(plugins/model-deepseek-v41)")


def test_qwen_uses_controller_capability_without_borrowing_provider_sources():
    source = (ROOT / "plugins/model-qwen3/CMakeLists.txt").read_text(encoding="utf-8")
    assert "qwen_execution_sources" not in source
    assert "pih_execution_default_impl" not in source
    assert "pih_packed_plan_values" in source
    sources = (ROOT / "plugins/model-qwen3/sources.cmake").read_text(encoding="utf-8")
    assert '"${CMAKE_CURRENT_LIST_DIR}/controller_client.cpp"' in sources
    for name in ("nvidia_qwen3_bf16_engine.cpp", "nvidia_qwen3_int4_engine.cpp",
                 "controller_packed_driver.cpp"):
        implementation = (ROOT / "plugins/model-qwen3" / name).read_text(encoding="utf-8")
        assert "ControllerRuntime::Create" not in implementation
        assert "ControllerRuntime&" not in implementation
    assert 'NOT IS_ABSOLUTE "${qwen_source}"' in source
    assert 'NOT EXISTS "${qwen_source}"' in source


def test_deepseek_plugin_owner_has_no_native_legacy_switch():
    owner = ROOT / "plugins/model-deepseek-v4-flash"
    paths = list(owner.rglob("*.cpp")) + list(owner.glob("*.cmake"))
    paths += [owner / "CMakeLists.txt"]
    paths += list((ROOT / "include/pih/backend/cuda").glob("nvidia_deepseek*.h"))
    paths += [ROOT / "include/pih/model/deepseek_stage_mapped_inventory.h"]
    for path in paths:
        assert "PIH_DEEPSEEK_NATIVE_PLUGIN" not in path.read_text(encoding="utf-8"), path


def test_deepseek_epoch_guard_depends_on_poller_not_controller_catalog():
    header = (ROOT / "include/pih/model/deepseek_artifact_epoch_guard.h").read_text(encoding="utf-8")
    engine = (ROOT / "plugins/model-deepseek-v4-flash/deepseek_engine.cpp").read_text(encoding="utf-8")
    assert "deepseek_controller_artifact_catalog" not in header
    assert "DmVeritySupervisorReceipt" not in header
    assert "poll(DeepSeekEngineArtifactPoller& poller)" in header
    assert "artifact_guard().poll(*artifact_poller_)" in engine


def test_deepseek_rank_runtime_has_no_native_legacy_switch():
    paths = (
        "plugins/model-deepseek-v4-flash/cuda/nvidia_deepseek_rank_runtime.cpp",
        "include/pih/backend/cuda/nvidia_deepseek_rank_runtime.h",
    )
    for path in paths:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "PIH_DEEPSEEK_NATIVE_PLUGIN" not in source
        for fallback in ("CudaAllocator::Create", "CudaMemoryCopier::Create",
                         "NvidiaPinnedHostAllocator::Create", "NvidiaRuntimeResourceDriver"):
            assert fallback not in source


def test_deepseek_bootstrap_requires_capability_catalog_and_runtime_factory():
    for path in (
        "plugins/model-deepseek-v4-flash/cuda/nvidia_deepseek_engine_bootstrap.cpp",
        "include/pih/backend/cuda/nvidia_deepseek_engine_bootstrap.h",
    ):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "PIH_DEEPSEEK_NATIVE_PLUGIN" not in source
        assert "DeepSeekControllerArtifactCatalog" not in source
        assert "poll_master_leases" not in source
        assert "NvidiaDeepSeekRankRuntime::Create(" not in source
        assert "NvidiaDeepSeekRankRuntimeFactory& runtime_factory" in source


@pytest.mark.parametrize("stem", (
    "nvidia_deepseek_h2d_runtime", "nvidia_deepseek_request_input_copy_operations",
    "nvidia_deepseek_recent_state_operations", "nvidia_deepseek_fixed_state_operations",
))
def test_deepseek_transfer_and_state_adapters_have_no_direct_cuda_fallback(stem):
    for path in (
        ROOT / "plugins/model-deepseek-v4-flash/cuda" / f"{stem}.cpp",
        ROOT / "include/pih/backend/cuda" / f"{stem}.h",
    ):
        source = path.read_text(encoding="utf-8")
        for retired in ("PIH_DEEPSEEK_NATIVE_PLUGIN", "Create()", "require_context",
                        "cudaMemcpyAsync", "cuEventRecord", "cuEventQuery",
                        "cudaEventRecord", "cudaEventQuery",
                        "#include <cuda", "cuda_driver_status.h"):
            assert retired not in source
