"""D685 bounded diagnostic, never imported by the serving implementation.

All tensors and pointer observations belong to ONE serialized inventory call.
No owner-epoch, model-ID, TTL, allocation or pinned-state cache is retained.
"""
import ast
import ctypes
import inspect
import textwrap
from pathlib import Path


def load_query(library: Path, torch):
    # PyDLL retains the GIL; do not allow another Python mutator inside the batch.
    native = ctypes.PyDLL(str(library.resolve(strict=True)))
    function = native.ieee_pin_batch
    function.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
                         ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_size_t)]
    function.restype = ctypes.c_int

    def query(entries):
        pointers = []
        for tensor, key in entries:
            if type(tensor) is not torch.Tensor or tensor.device.type != 'cpu':
                raise ValueError('batch witness requires exact native CPU tensors')
            if (type(key) is not tuple or len(key) != 2 or key[0] != 'cpu'
                    or type(key[1]) is not int or key[1] <= 0):
                raise ValueError('batch witness requires positive CPU storage base pointers')
            pointers.append(key[1])
        count = len(pointers)
        addresses = (ctypes.c_size_t * count)(*pointers)
        output = (ctypes.c_ubyte * count)()
        completed = ctypes.c_size_t()
        status = function(addresses, count, output, ctypes.byref(completed))
        if status != 0 or completed.value != count:
            raise RuntimeError(f'batched live CUDA pointer query failed: {status}; '
                               f'completed {completed.value}/{count}')
        return [bool(value) for value in output]
    return query


def inventory_witness(original, query, guard):
    """Exact-source clone: defer only per-view pin observations to one live loop.

    Every original tensor/storage/metadata validation remains. Provisional false
    pin flags are local and never published: actual results and alias agreement
    are checked before any physical graph is returned. Unknown source rejects.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    expected = ast.parse('pinned = bool(tensor.is_pinned())').body[0]
    class Replace(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(expected):
                self.count += 1
                return ast.parse('_pin_entries.append((tensor, key))\npinned = False').body
            return self.generic_visit(node)
    rewrite = Replace()
    tree = rewrite.visit(tree)
    if rewrite.count != 1 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('batch witness requires exactly one known per-view pin assignment')
    body = tree.body[0].body
    before = ast.dump(ast.parse('physical = list(allocations.values())').body[0])
    positions = [i for i, node in enumerate(body) if ast.dump(node) == before]
    if len(positions) != 1:
        raise RuntimeError('batch witness requires the exact physical publication boundary')
    body.insert(positions[0], ast.parse('_pin_finish(_pin_entries, allocations)').body[0])
    body[1:1] = ast.parse('_pin_guard()\n_pin_entries = []').body
    ast.fix_missing_locations(tree)

    def finish(entries, allocations):
        observed = query(entries)
        if len(observed) != len(entries) or any(type(value) is not bool for value in observed):
            raise RuntimeError('batch witness returned incomplete/nonboolean pin observations')
        seen = set()
        for (_, key), pinned in zip(entries, observed):
            allocation = allocations[key]
            if key in seen and allocation['pinned'] != pinned:
                raise ValueError('aliased native CPU storage has inconsistent capacity/pinning')
            allocation['pinned'] = pinned
            seen.add(key)
    namespace = {**original.__globals__, '_pin_guard': guard, '_pin_finish': finish}
    exec(compile(tree, '<diagnostic-batch-pin-inventory>', 'exec'), namespace)
    return namespace[original.__name__], ast.unparse(tree)
