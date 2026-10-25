"""D844 CPU component witness, not imported by serving code.

Reuse D685's one-call native witness/AST-clone method. Only the inventory's
second fresh-stat loop changes. Its original signature/envelope validation,
first scan, settlement restat, lock, publication and outputs stay in Python.
"""
import ast
from collections import namedtuple
import ctypes
import inspect
import os
from pathlib import Path
import textwrap


class _Record(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('device', 'inode', 'mode', 'nlink')] + [
        (name, ctypes.c_int64) for name in (
            'size', 'blocks', 'mtime_sec', 'mtime_nsec', 'ctime_sec', 'ctime_nsec')]


_Stat = namedtuple('_Stat', 'st_dev st_ino st_mode st_nlink st_size st_blocks st_mtime_ns st_ctime_ns')


def load_query(library):
    # CDLL releases the GIL once for the whole fresh syscall batch. The caller's
    # cooperative owner RLock remains held, as during the existing os.stat calls.
    native = ctypes.CDLL(str(library.resolve(strict=True)))
    size = native.ieee_file_stat_record_bytes
    size.argtypes, size.restype = [], ctypes.c_size_t
    if size() != ctypes.sizeof(_Record):
        raise RuntimeError('native file-stat record ABI mismatch')
    function = native.ieee_file_stat_batch
    function.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
                         ctypes.POINTER(_Record), ctypes.POINTER(ctypes.c_size_t)]
    function.restype = ctypes.c_int

    def query(paths):
        paths = list(paths)
        if any(type(path) is not str or '\0' in path for path in paths):
            raise ValueError('fresh file-stat batch requires normalized string paths without NUL')
        encoded = [os.fsencode(path) for path in paths]
        count = len(paths)
        addresses = (ctypes.c_char_p * count)(*encoded)
        output = (_Record * count)()
        completed = ctypes.c_size_t()
        status = function(addresses, count, output, ctypes.byref(completed))
        if (completed.value > count or (status == 0 and completed.value != count)
                or (status != 0 and completed.value >= count)):
            raise RuntimeError('native file-stat batch returned inconsistent completion')
        for i in range(completed.value):
            value = output[i]
            yield _Stat(value.device, value.inode, value.mode, value.nlink,
                value.size, value.blocks, value.mtime_sec*1_000_000_000 + value.mtime_nsec,
                value.ctime_sec*1_000_000_000 + value.ctime_nsec)
        if status:
            # Earlier signatures are validated before this error is delivered,
            # retaining ordered failure semantics and never exporting a partial view.
            raise OSError(status, os.strerror(status), paths[completed.value])
    return query


def inventory_witness(original, query):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    expected = ast.parse('for path, key, signature in observations:\n'
                         '    info = os.stat(path, follow_symlinks=False)\n').body[0]
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and ast.dump(node.target) == ast.dump(expected.target)
             and ast.dump(node.iter) == ast.dump(expected.iter)]
    if len(loops) != 1 or ast.dump(loops[0].body[0]) != ast.dump(expected.body[0]):
        raise RuntimeError('fresh-stat witness requires the exact existing second-pass boundary')
    loop = loops[0]
    loop.target = ast.parse('for (path, key, signature), info in ():\n    pass').body[0].target
    loop.iter = ast.parse('zip(observations, _batch_fresh_stats('
                         'row[0] for row in observations), strict=True)', mode='eval').body
    loop.body.pop(0)
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, '_batch_fresh_stats': query}
    exec(compile(tree, '<diagnostic-fresh-stat-batch>', 'exec'), namespace)
    return namespace[original.__name__], ast.unparse(tree)


class _TreeEntry(ctypes.Structure):
    _fields_ = [('path', ctypes.c_char_p), ('root', ctypes.c_size_t), ('value', _Record)]


def load_tree_scan(library):
    """D856 per-call native observation/validation; no cache or worker thread."""
    native = ctypes.CDLL(str(library.resolve(strict=True)))
    native.ieee_file_tree_entry_bytes.restype = ctypes.c_size_t
    if native.ieee_file_tree_entry_bytes() != ctypes.sizeof(_TreeEntry):
        raise RuntimeError('native file-tree entry ABI mismatch')
    scan = native.ieee_file_tree_scan
    scan.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
                     ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    scan.restype = ctypes.c_int
    count = native.ieee_file_tree_count
    count.argtypes, count.restype = [ctypes.c_void_p], ctypes.c_size_t
    entries = native.ieee_file_tree_entries
    entries.argtypes, entries.restype = [ctypes.c_void_p], ctypes.POINTER(_TreeEntry)
    release = native.ieee_file_tree_free
    release.argtypes, release.restype = [ctypes.c_void_p], None
    validate = native.ieee_file_tree_validate
    validate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int)]
    validate.restype = ctypes.c_int

    class Observation:
        def __init__(self, roots):
            self.roots = [(tier, str(Path(root))) for tier, root in roots.items()]
            if any('\0' in root for _, root in self.roots):
                raise ValueError('file-tree roots must not contain NUL')
            self.pointer = ctypes.c_void_p()
            self.closed = False
            self.first_stat_count = self.second_stat_count = 0

        def __enter__(self):
            try:
                encoded = [os.fsencode(root) for _, root in self.roots]
                failed = ctypes.c_size_t()
                status = scan((ctypes.c_char_p * len(encoded))(*encoded), len(encoded),
                              ctypes.byref(self.pointer), ctypes.byref(failed))
                length = count(self.pointer) if self.pointer else 0
                data = entries(self.pointer) if self.pointer else None
                if status:
                    path = os.fsdecode(data[failed.value].path) if failed.value < length else None
                    raise OSError(status, os.strerror(status), path)
                self.first_stat_count = length
                rows = []
                for index in range(length):
                    entry = data[index]
                    path, root = os.fsdecode(entry.path), entry.root
                    value = entry.value
                    info = _Stat(value.device, value.inode, value.mode, value.nlink,
                        value.size, value.blocks, value.mtime_sec*1_000_000_000 + value.mtime_nsec,
                        value.ctime_sec*1_000_000_000 + value.ctime_nsec)
                    # Python filename ordering, including surrogateescape. Sorting
                    # components (not whole strings) reconstructs the original DFS.
                    relative = path[len(self.roots[root][1].rstrip('/')) + 1:]
                    if self.roots[root][1] == '.':
                        relative = '' if path == '.' else path
                    key = () if path == self.roots[root][1] else tuple(relative.split('/'))
                    rows.append((root, key, index, path, info))
                rows.sort(key=lambda row: (row[0], row[1]))
                self.rows = rows
                self.order = (ctypes.c_size_t * length)(*(row[2] for row in rows))
                return self
            except BaseException:
                self.__exit__(None, None, None)
                raise

        def observations(self):
            for root, _, _, path, info in self.rows:
                yield self.roots[root][0], path, info

        def validate(self, writing_inodes, allocation_bounds):
            length = len(self.rows)
            flags = (ctypes.c_ubyte * length)()
            lower, upper = (ctypes.c_uint64 * length)(), (ctypes.c_uint64 * length)()
            for _, _, index, _, info in self.rows:
                key = (info.st_dev, info.st_ino)
                bound = allocation_bounds.get(key)
                if bound is not None:
                    low, high = bound['data_bytes'], bound['ceiling_bytes']
                    if not (type(low) is int and type(high) is int and 0 <= low <= high < 2**64):
                        raise ValueError('native allocation envelope requires uint64 byte bounds')
                    flags[index] |= 1
                    lower[index], upper[index] = low, high
                if key in writing_inodes:
                    flags[index] |= 2
            completed, reason = ctypes.c_size_t(), ctypes.c_int()
            status = validate(self.pointer, self.order, flags, lower, upper,
                              ctypes.byref(completed), ctypes.byref(reason))
            self.second_stat_count = completed.value + int(reason.value != 0)
            if reason.value == 2:
                raise RuntimeError('managed inode allocation exceeds its reserved envelope')
            if reason.value == 1:
                raise RuntimeError('managed inode changed while collecting footprint')
            if status:
                path = self.rows[completed.value][3] if completed.value < length else None
                raise OSError(status, os.strerror(status), path)
            if completed.value != length:
                raise RuntimeError('native file-tree validation incomplete')

        def __exit__(self, *unused):
            if not self.closed:
                release(self.pointer)
                self.pointer = ctypes.c_void_p()
                self.closed = True

    return Observation


def tree_inventory_witness(original, scan):
    """Clone only observation boundaries; keep accounting/settlement in Python."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    body = tree.body[0].body
    first = next(node for node in body if isinstance(node, ast.For) and
                 ast.unparse(node.target) == '(tier, root)')
    inner = first.body[-1]
    if not isinstance(inner, ast.While) or ast.unparse(inner.body[1]) != 'info = os.stat(path, follow_symlinks=False)':
        raise RuntimeError('tree witness requires the existing fresh traversal boundary')
    loop = ast.parse('for tier, path, info in session.observations():\n    pass').body[0]
    if not (isinstance(inner.body[-1], ast.If) and ast.unparse(inner.body[-1].test) == "kind == 'directory'"):
        raise RuntimeError('tree witness requires the existing directory boundary')
    loop.body = [node for node in inner.body[2:-1] if not (
        isinstance(node, ast.Expr) and ast.unparse(node).startswith('observations.append('))]
    second = next(node for node in body if isinstance(node, ast.For) and
                  ast.unparse(node.target) == '(path, key, signature)')
    start, end = body.index(first), body.index(second)
    if end != start + 1:
        raise RuntimeError('tree witness requires adjacent original observation passes')
    wrapper = ast.parse('with _scan_file_tree(roots) as session:\n    pass').body[0]
    wrapper.body = [loop, ast.parse('session.validate(writing_inodes, allocation_bounds)').body[0],
                    *body[end+1:]]
    body[start:] = [wrapper]
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, '_scan_file_tree': scan}
    exec(compile(tree, '<diagnostic-native-file-tree>', 'exec'), namespace)
    return namespace[original.__name__], ast.unparse(tree)
