from __future__ import annotations

import json
from pathlib import Path

import httpx

from harbor_hf.provider_models import (
    ExplicitProviderRoute,
    ProviderLimits,
    ProviderTarget,
)
from harbor_hf.provider_proxy import ProviderEvidenceProxy


def test_proxy_forwards_stream_and_records_content_free_provider_evidence(
    tmp_path: Path,
) -> None:
    observed: list[dict[str, object]] = []
    stream = (
        b'data: {"id":"response-one","model":"org/model",'
        b'"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":11,"completion_tokens":4,'
        b'"total_tokens":15}}\n\n'
        b"data: [DONE]\n\n"
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(
            {
                "authorization": request.headers.get("authorization"),
                "payload": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "provider-request-one",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "99",
            },
            stream=httpx.ByteStream(stream),
        )

    target = ProviderTarget(
        id="provider-one",
        model="org/model",
        routing=ExplicitProviderRoute(provider="groq"),
        limits=ProviderLimits(max_attempts=2),
        parameters={"temperature": 0},
    )
    evidence_path = tmp_path / "provider-requests.jsonl"
    upstream_client = httpx.Client(transport=httpx.MockTransport(upstream))
    proxy = ProviderEvidenceProxy(
        target,
        token="secret-token",
        evidence_path=evidence_path,
        client=upstream_client,
    )
    base_url = proxy.start()
    try:
        response = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "ignored/model",
                "messages": [{"role": "user", "content": "private benchmark prompt"}],
                "stream": True,
            },
            timeout=5,
        )
    finally:
        proxy.close()
        upstream_client.close()

    assert response.status_code == 200
    assert response.content == stream
    assert observed == [
        {
            "authorization": "Bearer secret-token",
            "payload": {
                "messages": [{"content": "private benchmark prompt", "role": "user"}],
                "model": "org/model:groq",
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": 0,
            },
        }
    ]
    raw_evidence = evidence_path.read_text(encoding="utf-8")
    assert "private benchmark prompt" not in raw_evidence
    assert "answer" not in raw_evidence
    assert "secret-token" not in raw_evidence
    evidence = json.loads(raw_evidence)
    assert evidence["status"] == "succeeded"
    assert evidence["evidence"]["request"] == {
        "message_count": 1,
        "provider_request_id": {
            "detail": None,
            "status": "observed",
            "value": "provider-request-one",
        },
        "request_id": "provider-1",
        "streaming": True,
        "tool_count": 0,
    }
    assert evidence["evidence"]["usage"] == {
        "input_tokens": {"detail": None, "status": "observed", "value": 11},
        "output_tokens": {"detail": None, "status": "observed", "value": 4},
        "total_tokens": {"detail": None, "status": "observed", "value": 15},
    }
