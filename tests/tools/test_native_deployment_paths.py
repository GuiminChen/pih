"""Path admission only; never invoke CMake, a worker or a model."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_path_deploy", Path(__file__).resolve().parents[2] / "deploy/native.py")
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(DEPLOY, "ROOT", source)
    return source, model


def test_sibling_outputs_allowed(layout):
    source, model = layout
    build, bundle = source / "out/build", source / "out/bundle"
    assert DEPLOY.validate_deployment_paths(build, bundle, model) == (build, bundle)


def test_overlapping_outputs_rejected(layout):
    source, model = layout
    build = source / "out/build"
    for bundle in (build, build / "install", build.parent):
        with pytest.raises(ValueError, match="disjoint"):
            DEPLOY.validate_deployment_paths(build, bundle, model)


def test_broad_output_rejected(layout):
    source, model = layout
    for output in (source, source.parent, Path(source.anchor)):
        with pytest.raises(ValueError, match="dedicated"):
            DEPLOY.validate_deployment_paths(output, source / "out/bundle", model)


def test_install_cannot_overlap_model(layout):
    source, model = layout
    for bundle in (model, model / "install", model.parent):
        with pytest.raises(ValueError):
            DEPLOY.validate_deployment_paths(source / "out/build", bundle, model)


def test_file_output_rejected(layout):
    source, model = layout
    bundle = source / "bundle-file"
    bundle.touch()
    with pytest.raises(ValueError, match="directories"):
        DEPLOY.validate_deployment_paths(source / "out/build", bundle, model)


def test_build_preserves_existing_bundle(layout, monkeypatch):
    source, _ = layout
    bundle = source / "bundle"
    bundle.mkdir()
    owned = bundle / "user-file"
    owned.write_text("preserve")
    monkeypatch.setattr(DEPLOY.platform, "system", lambda: "Linux")
    monkeypatch.setattr(DEPLOY, "run", lambda _: pytest.fail("must reject before building"))
    with pytest.raises(ValueError, match="new bundle"):
        DEPLOY.build(source / "build", bundle, 1, "rtx4090d")
    assert owned.read_text() == "preserve"
