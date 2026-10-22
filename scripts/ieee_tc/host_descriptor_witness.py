"""D747 same-call fused metadata witness; never imported by production.

Unlike D685, this has no batch of retained tensor/tuple entries, direct CUDA
query, delayed pin check or provisional graph. Every descriptor is current and
fully validated before the original alias/accounting code consumes it.
"""
import ast
import importlib.util
import inspect
import textwrap


def load_descriptor(library):
    spec = importlib.util.spec_from_file_location(
        '_ieee_native_host_descriptor', library.resolve(strict=True))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read


def inventory_witness(original, descriptor, guard):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('descriptor witness requires one inventory function')
    adds = [node for node in tree.body[0].body
            if isinstance(node, ast.FunctionDef) and node.name == 'add']
    if len(adds) != 1:
        raise RuntimeError('descriptor witness requires one known tensor visitor')
    prefix = ast.parse('''
if (not torch.is_tensor(tensor) or tensor.device.type != 'cpu'
        or tensor.layout != torch.strided or len(tensor.shape) != 2
        or tensor.numel() <= 0):
    raise ValueError(f'expected nonempty dense native CPU LoRA tensor: {name}')
storage = tensor.untyped_storage()
pointer, capacity = int(storage.data_ptr()), int(storage.nbytes())
if pointer <= 0 or capacity <= 0:
    raise ValueError(f'empty native CPU LoRA storage: {name}')
key = (str(tensor.device), pointer)
pinned = bool(tensor.is_pinned())
''').body
    # D748 preserves the exact original path under an explicit descriptor
    # selection. Accept that known structure without weakening the source guard.
    body = adds[0].body
    count = len(prefix)
    if (isinstance(body[0], ast.If)
            and ast.dump(body[0].test) == ast.dump(ast.parse('descriptor is None', mode='eval').body)):
        native = ast.parse("pointer, capacity, pinned = descriptor(tensor, name)\nkey = ('cpu', pointer)").body
        if [ast.dump(n) for n in body[0].orelse] != [ast.dump(n) for n in native]:
            raise RuntimeError('descriptor witness native visitor source changed')
        checked = body[0].body
        count = 1
    else:
        checked = body[:count]
    if [ast.dump(n) for n in checked] != [ast.dump(n) for n in prefix]:
        raise RuntimeError('descriptor witness tensor visitor source changed')
    adds[0].body[:count] = ast.parse('''
pointer, capacity, pinned = _fresh_descriptor(tensor, name)
key = ('cpu', pointer)
''').body
    tree.body[0].body.insert(1, ast.parse('_descriptor_guard()').body[0])
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, '_fresh_descriptor': descriptor,
                 '_descriptor_guard': guard}
    exec(compile(tree, '<diagnostic-fused-host-descriptor>', 'exec'), namespace)
    return namespace[original.__name__], ast.unparse(tree)
