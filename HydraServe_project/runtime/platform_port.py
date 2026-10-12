"""Source-pinned lifecycle injection for HydraServe's original controller.

This renders two small source overlays in memory; it neither changes the clean
vendor checkout nor starts containers. A concrete, qualified platform is still
required. No scheduler, placement, network model, or consolidation policy is
replaced. Without a platform, the original Kubernetes path remains available.
"""
from hashlib import sha256
from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol, Tuple


NATIVE_COMMIT = "8ae605de354ccfa2e9095514cdb4a9e9c56aa56b"
SOURCE_DIR = "scripts/kubernetes/vllm/src"
SOURCE_SHA256 = {
    "resource_manager.py": "3a88b5f784eb089fd31b7378f688c356b5c57cad837ba1f334c01b4f8ab56d52",
    "scheduler.py": "7c1d59a91fc12e92dcdb7aad3179f27281075cf21ab83f109b8901ba9f49eea5",
    "algs.py": "12dd34c2cea578b64dcf7ed6f35efdff4b6a6fcd2a87c4d33191300ccae3ece1",
}


class LifecyclePlatform(Protocol):
    """Only infrastructure, never fake Pod readiness or scheduling policy.

    Inventory is physical-node -> (GPU count, per-GPU GB, Gbit/s, host GB),
    matching native units. UUID identities and live resource evidence belong in
    the concrete platform manifest. One four-GPU machine is ONE inventory entry.
    Storage addresses must identify the actual project-local storage service.
    create_worker must establish restrictions before any worker/model process.
    An IP is an address, NOT model readiness: native initialize_pods still calls
    and awaits /init. Deletion is confirmed only after actual owned workers and
    GPU holders have exited. A CLI acknowledgement is insufficient.
    """

    def inventory(self) -> Mapping[str, Tuple[int, float, float, float]]: ...

    def storage_ip(self, node: str) -> str: ...

    async def create_worker(self, name: str, image: str, env: Dict[str, str], *,
                            node: str, gpu_id: int, nas_path: str,
                            gpu: Tuple[float, int]) -> None: ...

    async def worker_ip(self, name: str) -> Optional[str]: ...

    async def delete_worker(self, name: str) -> None: ...

    async def worker_deleted(self, name: str) -> bool: ...


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("native platform splice must have exactly one match")
    return source.replace(old, new, 1)


def render_overlays(native_root: Path) -> Dict[str, bytes]:
    """Fail closed on all three input identities; return only changed files.

    The caller must write to a new per-run overlay, save its SHA/patch and qualify
    the actual installed source. Calling this function is NOT an installation.
    """
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        raw = (Path(native_root) / SOURCE_DIR / name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError("unqualified native source: " + name)
        sources[name] = raw.decode("utf-8")
    source = sources["resource_manager.py"]
    edits = [
        ("def __init__(self, core_api, apps_api):\n",
         "def __init__(self, core_api, apps_api, *, platform=None):\n"
         "        self.platform = platform\n"),
        ("        nodes = get_node_list_with_resource(self.core_api)\n",
         "        nodes = (self.platform.inventory() if self.platform is not None\n"
         "                 else get_node_list_with_resource(self.core_api))\n"),
        ("    def get_storage_server_ip(self):\n",
         "    def get_storage_server_ip(self):\n"
         "        if self.platform is not None:\n"
         "            self.storage_server_ip = [self.platform.storage_ip(node)\n"
         "                                      for node in self.node_list]\n"
         "            return\n"),
        ("            delete_deployment(self.apps_api, resource_usage.pod_name)\n",
         "            if self.platform is None:\n"
         "                delete_deployment(self.apps_api, resource_usage.pod_name)\n"
         "            else:\n"
         "                await self.platform.delete_worker(resource_usage.pod_name)\n"),
        ("            while not check_deployment_fin(self.apps_api, resource_usage.pod_name):\n",
         "            while not (check_deployment_fin(self.apps_api, resource_usage.pod_name)\n"
         "                       if self.platform is None else\n"
         "                       await self.platform.worker_deleted(resource_usage.pod_name)):\n"),
        ("            create_deployment(self.apps_api, pod_name, image, env,\n"
         "                              node=self.node_list[node_id],\n"
         "                              nas_path=nas_path,\n"
         "                              gpu=(gpu_usage, core_percentage))\n",
         "            if self.platform is None:\n"
         "                create_deployment(self.apps_api, pod_name, image, env,\n"
         "                                  node=self.node_list[node_id],\n"
         "                                  nas_path=nas_path,\n"
         "                                  gpu=(gpu_usage, core_percentage))\n"
         "            else:\n"
         "                await self.platform.create_worker(\n"
         "                    pod_name, image, dict(env), node=self.node_list[node_id],\n"
         "                    gpu_id=gpu_id, nas_path=nas_path,\n"
         "                    gpu=(gpu_usage, core_percentage))\n"),
        ("        # Get ip list\n",
         "        if self.platform is not None:\n"
         "            ip_list = [None] * allocated_resource.num_nodes\n"
         "            while any(ip is None for ip in ip_list):\n"
         "                await asyncio.sleep(0.1)\n"
         "                for rank, name in enumerate(pods_name):\n"
         "                    if ip_list[rank] is None:\n"
         "                        ip_list[rank] = await self.platform.worker_ip(name)\n"
         "            return ip_list\n"
         "\n"
         "        # Get ip list\n"),
    ]
    for old, new in edits:
        source = _replace_once(source, old, new)
    scheduler = _replace_once(sources["scheduler.py"],
        "    def __init__(self, core_api, apps_api, models: Dict[str, Dict[str, Tuple[int, int, int, int, int]]]):\n"
        "        self.resource_manager = ResourceManager(core_api, apps_api)\n",
        "    def __init__(self, core_api, apps_api, models: Dict[str, Dict[str, Tuple[int, int, int, int, int]]], *, platform=None):\n"
        "        self.resource_manager = ResourceManager(core_api, apps_api, platform=platform)\n")
    overlays = {"resource_manager.py": source, "scheduler.py": scheduler}
    for name, text in overlays.items():
        compile(text, name, "exec")
    return {SOURCE_DIR + "/" + name: text.encode("utf-8")
            for name, text in overlays.items()}
