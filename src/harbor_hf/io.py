from pathlib import Path

import yaml
from pydantic import ValidationError

from harbor_hf.models import ExperimentSpec


class ManifestError(ValueError):
    """Raised when an experiment manifest cannot be loaded or validated."""


def load_experiment(path: Path) -> ExperimentSpec:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    return load_experiment_bytes(content, source=str(path))


def load_experiment_bytes(content: bytes, *, source: str) -> ExperimentSpec:
    """Validate an experiment manifest read from a remote control snapshot."""
    try:
        raw = yaml.safe_load(content)
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot read {source}: {error}") from error

    if not isinstance(raw, dict):
        raise ManifestError(f"{source} must contain a YAML object")

    try:
        return ExperimentSpec.model_validate(raw)
    except ValidationError as error:
        raise ManifestError(str(error)) from error
