"""Pinned, opt-in address adaptation for multiple native workers on one host.

No fake physical nodes, proxy inference endpoint, scheduling or tensor changes.
The platform must validate and reserve each address before container creation.
This renderer does not install or execute the resulting source.
"""
from hashlib import sha256
from ipaddress import IPv4Address
from pathlib import Path
import re


SOURCE_SHA256 = {
    "vllm/executor/gpu_executor.py": "8f7ff36d534b74789b06f7afea40a14bd9bfefac803566275074239c8d9ddebf",
    "scripts/image/app.py": "cab8bc3195159a51da0c4e8ca332a38b049f65891e49cdcd2534fe984235717f",
    "scripts/kubernetes/vllm/storage_server/local_server.py": "dbf245782059b89a49254b557fe7634e76ce4030fe6f1d84c6cc467182556ca1",
    "scripts/kubernetes/vllm/storage_server/remote_server.py": "f457af0b70335df239dcfbda70a4281a29afc2e6aace684787c527048cbbda6c",
}


def address_environment(address: str) -> dict:
    """Private IPv4 loopback only; not a substitute for actual bind/owner checks."""
    parsed = IPv4Address(address)
    if not parsed.is_loopback or str(parsed) != address:
        raise ValueError("HydraServe local worker requires canonical loopback IPv4")
    return {"HYDRASERVE_BIND_IP": address, "VLLM_HOST_IP": address,
            "HYDRASERVE_BIND_SOURCE_IP": "1"}


def render_overlays(native_root: Path) -> dict:
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        raw = (Path(native_root)/name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError("unqualified native source: " + name)
        sources[name] = raw.decode("utf-8")
    key = "vllm/executor/gpu_executor.py"
    pattern = r"(?m)^([ \t]*)client_socket\.setsockopt\(socket\.SOL_SOCKET, socket\.SO_PRIORITY, 6\)\n"

    def insert(match):
        indent = match.group(1)
        return (match.group(0) + indent + 'if os.getenv("HYDRASERVE_BIND_SOURCE_IP") == "1":\n'
                + indent + '    client_socket.bind((ip_list[my_rank], 0))\n')

    sources[key], count = re.subn(pattern, insert, sources[key])
    if count != 3:
        raise ValueError("unexpected native pipeline client socket sites")
    edits = {
        "scripts/image/app.py": ("host='0.0.0.0',", "host=os.getenv('HYDRASERVE_BIND_IP', '0.0.0.0'),"),
        "scripts/kubernetes/vllm/storage_server/local_server.py": (
            "asyncio.start_server(handler, '0.0.0.0', 6666)",
            "asyncio.start_server(handler, os.getenv('HYDRASERVE_BIND_IP', '0.0.0.0'), 6666)"),
        "scripts/kubernetes/vllm/storage_server/remote_server.py": (
            "socketserver.ForkingTCPServer(('0.0.0.0', 8888), MyTCPHandler)",
            "socketserver.ForkingTCPServer((os.getenv('HYDRASERVE_BIND_IP', '0.0.0.0'), 8888), MyTCPHandler)"),
    }
    for name, (old, new) in edits.items():
        if sources[name].count(old) != 1:
            raise ValueError("unexpected native listening site: " + name)
        sources[name] = sources[name].replace(old, new, 1)
    for name, source in sources.items():
        compile(source, name, "exec")
    return {name:source.encode("utf-8") for name, source in sources.items()}
