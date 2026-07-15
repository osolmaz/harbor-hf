from __future__ import annotations

from collections.abc import Mapping

from harbor_hf.harbor_native_bundle import HarborNativeBundle
from harbor_hf.publication_envelope import PublicationEnvelope
from harbor_hf.results import CatalogRow, ResultProjection

PUBLICATION_SCHEMA_MODELS = {
    "harbor-native-bundle-v1alpha1.schema.json": HarborNativeBundle,
    "publication-envelope-v1.schema.json": PublicationEnvelope,
    "result-projection-v1.schema.json": ResultProjection,
    "result-catalog-v1.schema.json": CatalogRow,
}


def publication_schema_documents() -> Mapping[str, dict[str, object]]:
    """Return the canonical JSON Schemas checked into the repository."""
    return {
        filename: model.model_json_schema()
        for filename, model in PUBLICATION_SCHEMA_MODELS.items()
    }
