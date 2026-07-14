from pathlib import Path

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from harbor_hf.models import ExperimentSpec


class ManifestError(ValueError):
    """Raised when an experiment manifest cannot be loaded or validated."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    explicit_keys: set[object] = set()
    for key_node, _value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in explicit_keys
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        explicit_keys.add(key)
    return super(_UniqueKeyLoader, loader).construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_experiment(path: Path) -> ExperimentSpec:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    return load_experiment_bytes(content, source=str(path))


def load_experiment_bytes(content: bytes, *, source: str) -> ExperimentSpec:
    """Validate an experiment manifest read from a remote control snapshot."""
    try:
        raw = yaml.load(content, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot read {source}: {error}") from error

    if not isinstance(raw, dict):
        raise ManifestError(f"{source} must contain a YAML object")

    try:
        return ExperimentSpec.model_validate(raw)
    except ValidationError as error:
        raise ManifestError(str(error)) from error
