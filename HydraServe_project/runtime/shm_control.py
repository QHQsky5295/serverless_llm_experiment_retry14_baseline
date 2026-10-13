"""Native SHM control lease whose lifetime does not belong to a transient loop.

HydraServe opens at import under asyncio.run, then releases during uvicorn init.
An asyncio StreamWriter belongs to the first, already-closed loop. Use an owned
nonblocking socket with loop.sock_* only for pending I/O, retaining the original
wire header and EOF release protocol. No new loader/cache/scheduling policy.
"""
import ast
import asyncio
from hashlib import sha256
from pathlib import Path
import socket

APP = 'scripts/image/app.py'
APP_SHA = 'cab8bc3195159a51da0c4e8ca332a38b049f65891e49cdcd2534fe984235717f'


class NativeShmLease:
    def __init__(self):
        self._socket = None

    async def begin(self, address, header_bytes, port=6666):
        if self._socket is not None:
            raise RuntimeError('native SHM lease already active')
        if not isinstance(header_bytes, bytes) or not header_bytes:
            raise ValueError('native SHM header must be nonempty bytes')
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setblocking(False)
        self._socket = connection
        loop = asyncio.get_running_loop()
        async def exact(size):
            chunks = bytearray()
            while len(chunks) < size:
                part = await loop.sock_recv(connection, size-len(chunks))
                if not part:
                    raise EOFError('native SHM descriptor truncated')
                chunks.extend(part)
            return bytes(chunks)
        try:
            await loop.sock_connect(connection, (address, port))
            await loop.sock_sendall(connection, len(header_bytes).to_bytes(8,'little')+header_bytes)
            address_bytes, size_bytes = await exact(8), await exact(8)
            # Preserve native unframed <=50-byte name response semantics.
            # This does not claim to repair/generalize that wire protocol.
            name_bytes = await loop.sock_recv(connection, 50)
            address_value = int.from_bytes(address_bytes,'little',signed=True)
            size_value = int.from_bytes(size_bytes,'little',signed=True)
            name = name_bytes.decode('utf-8')
            if not name or address_value < 0 or size_value <= 0:
                raise ValueError('invalid native SHM descriptor')
            return name, address_value, size_value
        except BaseException:
            self.close()
            raise

    def close(self):
        connection, self._socket = self._socket, None
        if connection is None:
            return False
        connection.close()  # EOF on original connection; no event-loop callback.
        return True


START = '''async def start_load_model_shm(model_id: str, pp_rank: int, pp_size: int, is_dest: bool):
    stime = time.time()
    server_ip = os.getenv('STORAGE_IP')
    req_header = {"model": model_id, "pp_rank": pp_rank, "pp_size": pp_size, "is_dest": is_dest, "pre_load": "no"}
    req_header_bytes = json.dumps(req_header).encode('utf-8')
    shm_name, shm_addr, shm_size = await shm_control_lease.begin(server_ip, req_header_bytes)
    try:
        torch.ops.download_model.start_load_shm(shm_name, shm_addr, shm_size)
    except BaseException:
        shm_control_lease.close()
        raise
    print(f"start_load_model_shm: start_load_shm time cost = {time.time() - stime} seconds")
'''
END = '''async def end_load_model_shm():
    stime = time.time()
    shm_control_lease.close()
    print(f"end_load_model_shm: close connection time cost = {time.time() - stime} seconds")
'''


def render_app(source, native_root):
    """Replace only control-connection ownership on a source-pinned app overlay."""
    raw=(Path(native_root)/APP).read_bytes()
    if sha256(raw).hexdigest()!=APP_SHA:
        raise ValueError('native SHM app source drift')
    originals={n.name:n for n in ast.parse(raw).body if isinstance(n,ast.AsyncFunctionDef)}
    tree=ast.parse(source); changes=[]
    for name,replacement in (('start_load_model_shm',START),('end_load_model_shm',END)):
        current=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name==name)
        if ast.dump(current)!=ast.dump(originals[name]):
            raise ValueError('native SHM function already changed: '+name)
        changes.append((current.lineno-1,current.end_lineno,replacement))
    lines=source.splitlines(keepends=True)
    for start,end,replacement in sorted(changes,reverse=True):
        lines[start:end]=[replacement]
    result=''.join(lines)
    if result.count('writer = None\n')!=1:
        raise ValueError('native SHM owner declaration drift')
    result=result.replace('writer = None\n',
        'from HydraServe_project.runtime.shm_control import NativeShmLease\n'
        'shm_control_lease = NativeShmLease()\n',1)
    compile(result,APP,'exec')
    return result
