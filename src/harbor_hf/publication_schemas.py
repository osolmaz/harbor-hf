from __future__ import annotations

from collections.abc import Mapping

from harbor_hf.harbor_native_bundle import HarborNativeBundle
from harbor_hf.publication_envelope import PublicationEnvelopeV2
from harbor_hf.results import CatalogRowV2, ResultProjectionV2

PUBLICATION_SCHEMA_MODELS = {
    "harbor-native-bundle-v1alpha1.schema.json": HarborNativeBundle,
    "publication-envelope-v2.schema.json": PublicationEnvelopeV2,
    "result-projection-v2.schema.json": ResultProjectionV2,
    "result-catalog-v2.schema.json": CatalogRowV2,
}


def publication_schema_documents() -> Mapping[str, dict[str, object]]:
    """Return the canonical JSON Schemas checked into the repository."""
    return {
        filename: model.model_json_schema()
        for filename, model in PUBLICATION_SCHEMA_MODELS.items()
    }
