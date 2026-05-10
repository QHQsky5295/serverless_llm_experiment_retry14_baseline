from __future__ import annotations

import threading
from pathlib import Path

from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
from remote_artifact_node.server import ArtifactHandler, ArtifactServer


def test_http_artifact_store_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    adapter = root / "demo_lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path":"demo"}\n')
    (adapter / "adapter_model.safetensors").write_text("demo\n")

    server = ArtifactServer(("127.0.0.1", 0), ArtifactHandler, root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        client = HttpArtifactStoreClient(endpoint=endpoint, timeout_s=10)
        assert client.health()["ok"] is True
        assert client.list_artifacts() == ["demo_lora"]
        target = tmp_path / "fetch" / "demo_lora"
        ok, _elapsed_ms, size_bytes = client.download_artifact("demo_lora", str(target))
        assert ok
        assert size_bytes > 0
        assert (target / "adapter_config.json").exists()
        assert (target / "adapter_model.safetensors").exists()
    finally:
        server.shutdown()
        server.server_close()
