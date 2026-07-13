from pathlib import Path

import yaml
from pydantic import ValidationError

from harbor_hf.models import ExperimentSpec


class ManifestError(ValueError):
    """Raised when an experiment manifest cannot be loaded or validated."""


def load_experiment(path: Path) -> ExperimentSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot read {path}: {error}") from error

    if not isinstance(raw, dict):
        raise ManifestError(f"{path} must contain a YAML object")

    try:
        return ExperimentSpec.model_validate(raw)
    except ValidationError as error:
        raise ManifestError(str(error)) from error
