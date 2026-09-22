"""Static cutover guards, not native converter or inference qualification."""

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_old_package_initializer_has_no_executable_api() -> None:
    initializer = ROOT / "python/pih/__init__.py"
    if not initializer.exists():
        return
    tree = ast.parse(initializer.read_text(encoding="utf-8"))
    assert len(tree.body) == 1
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)


def test_removed_conversion_wrappers_stay_removed() -> None:
    for name in ("artifact_receipt.py", "int4_artifact_receipt.py",
                 "convert_qwen3_int4_cli.py"):
        assert not (ROOT / "python/pih" / name).exists(), name


def test_removed_python_service_hosts_stay_removed() -> None:
    for name in ("qwen_int4_service_host.py", "deepseek_service_host.py",
                 "qwen_int4_service_assembly.py", "deepseek_service_assembly.py",
                 "qwen_int4_service_runtime.py", "deepseek_service_runtime.py",
                 "deepseek_nonstream_executor.py", "nonstream_executor.py"):
        assert not (ROOT / "python/pih" / name).exists(), name


def test_retired_python_service_collectors_stay_removed() -> None:
    for path in (
        "tests/hardware/qwen_m2_native_service_smoke.py",
        "tools/run_qwen_m2_target_suite.py",
    ):
        assert not (ROOT / path).exists(), path


def test_retired_python_connection_pipeline_stays_removed() -> None:
    for name in (
        "application_streaming_session", "application_request_runner",
        "accepted_socket_session", "fixed_connection_driver",
        "fixed_connection_loop", "http_connection_state", "bounded_socket_writer",
        "linux_asyncio_socket", "authenticated_connection",
        "chat_streaming_session", "sse_write_bridge",
    ):
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name


def test_retired_python_async_engines_stay_removed() -> None:
    for name in ("async_engine.py", "async_deepseek_engine.py"):
        assert not (ROOT / "python/pih" / name).exists(), name
    for name in ("test_async_engine.py", "test_async_deepseek_engine.py",
                 "test_request_handle.py", "test_import.py"):
        assert not (ROOT / "tests/python" / name).exists(), name


def test_retired_python_engine_and_service_router_stay_removed() -> None:
    for name in ("engine", "model", "deepseek_engine", "service_snapshot",
                 "service_router", "ingress_gate", "ingress_transaction"):
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
    for name in ("test_engine", "test_model", "test_deepseek_engine",
                 "test_service_snapshot", "test_service_router",
                 "test_ingress_gate", "test_ingress_transaction"):
        assert not (ROOT / "tests/python" / f"{name}.py").exists(), name


def test_retired_python_http_listener_and_serializers_stay_removed() -> None:
    for name in ("linux_accepted_server", "http_framing",
                 "http_header_receiver", "http_body_receiver",
                 "openai_sse", "openai_response"):
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
        assert not (ROOT / "tests/python" / f"test_{name}.py").exists(), name


def test_retired_python_service_request_chain_stays_removed() -> None:
    modules = ("deepseek_service_scheduler", "service_admission",
               "async_tokenizer_admission", "tokenizer_admission_bridge",
               "tokenizer_worker_pool", "qwen_worker_execute", "tokenizer_source",
               "openai_request", "prompt_admission", "sampling")
    for name in modules:
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
        assert not (ROOT / "tests/python" / f"test_{name}.py").exists(), name
    assert not (ROOT / "python/pih/stream.py").exists()
    assert not (ROOT / "tests/python/test_stream_queue.py").exists()


def test_retired_python_http_policy_chain_stays_removed() -> None:
    modules = ("bounded_json_body", "connection_error_wire", "http_errors",
               "http_response_envelope", "ingress_auth", "ingress_credits",
               "operational_credits", "operational_endpoint_inventory",
               "operational_schema", "operational_service_envelope", "peer_auth",
               "linux_peercred", "listener_manifest", "writer_backing",
               "writer_credits")
    for name in modules:
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
        assert not (ROOT / "tests/python" / f"test_{name}.py").exists(), name


def test_retired_python_controller_and_output_chain_stays_removed() -> None:
    modules = (
        "deepseek_controller", "engine_state", "supervisor_shutdown",
        "request_identity", "tokenizer_ring", "tool_call_projection",
        "qwen_packed_output_lane", "deepseek_incremental_parser",
        "deepseek_output_event_buffer", "deepseek_output_sequence",
        "incremental_detokenizer", "qwen_output_sequence",
        "qwen_thinking_parser", "stop_matcher",
    )
    for name in modules:
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
        assert not (ROOT / "tests/python" / f"test_{name}.py").exists(), name
    for name in ("test_supervisor_shutdown_bootstrap.py",
                 "test_supervisor_shutdown_operation.py"):
        assert not (ROOT / "tests/python" / name).exists(), name


def test_retired_python_placement_and_preflight_chain_stays_removed() -> None:
    modules = (
        "catalog_placement", "runtime_admission", "service_capacity_instance",
        "service_fd_preflight", "service_process_preflight",
        "service_resource_preflight", "_capabilities",
    )
    for name in modules:
        assert not (ROOT / "python/pih" / f"{name}.py").exists(), name
        assert not (ROOT / "tests/python" / f"test_{name}.py").exists(), name


def test_removed_native_binding_has_no_python_runtime_compiler() -> None:
    assert not (ROOT / "python/pih/runtime_profile.py").exists()
    assert not (ROOT / "tests/python/test_runtime_profile.py").exists()
    assert not (ROOT / "tests/python/test_qwen_native_paired_delta_buffer.py").exists()
    for source in (ROOT / "python/pih").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "from . import _pih" not in text, source.name
        assert "from pih import _pih" not in text, source.name
    for source in (ROOT / "tests/python").glob("test_*.py"):
        text = source.read_text(encoding="utf-8")
        assert "pih._pih" not in text, source.name
        assert "from pih import _pih" not in text, source.name


def test_distribution_exposes_only_http_client_package() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["tool"]["setuptools"]["packages"] == ["pih_client"]
    assert metadata["project"]["scripts"] == {
        "pih-client": "pih_client.__main__:main",
        "pih-preflight-target-host": "pih_client.preflight:main",
    }
