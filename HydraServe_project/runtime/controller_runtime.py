"""Load the full original controller against the qualified lifecycle port.

Only make Kubernetes SDK imports local to the original Kubernetes functions.
The injected platform avoids those functions; calling one still needs the real
SDK. No stub cluster, HTTP/TCP replacement, placement or scheduler algorithm.
Use one controller in a dedicated process: original modules use flat imports.
"""
import ast
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys

from .native_overlay_entry import SourceOverlay
from .platform_port import SOURCE_DIR, SOURCE_SHA256
from .request_contract import render_overlays as request_overlays, SOURCE_SHA, ENGINE

NATIVE_SHA={
    **{name[:-3]:digest for name,digest in SOURCE_SHA256.items()},
    'engine': SOURCE_SHA[ENGINE],
    'utils':'74acf604e1f86e6365aa4a64c1076b3ea1d01da38569904ae11beea7bd219386',
    'ImageInfo':'f6e9b4a5e3cf1c22dd2be36b3800f80034dae07a71864c548f0041a077a035a2',
    'ECSInstanceInfo':'ae9ec759425d20e10b889cd0883a4e3e3aedfa1f4bc67eba8c4f6acd420fe76c',
    'cache_manager':'ae28f477f64f014ba961b3ecfe480d2336bc6e3c792afd98e7a06d84e5293f06',
}
ALIASES={name:SOURCE_DIR+'/'+name+'.py' for name in NATIVE_SHA}
KUBE_FUNCTIONS={'create_deployment','delete_deployment','delete_all_deployments','create_job','delete_job'}


def render_controller_sources(native_root):
    """Manifest-ready full modules; unchanged HTTP/TCP/algorithm source retained."""
    root=Path(native_root); overlays=request_overlays(root); result={}
    for name,relative in ALIASES.items():
        raw=(root/relative).read_bytes()
        if sha256(raw).hexdigest()!=NATIVE_SHA[name]:
            raise ValueError('unqualified native controller source: '+name)
        result[relative]=overlays.get(relative,raw)
    source=result[ALIASES['utils']].decode()
    tree=ast.parse(source)
    uses={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
          and any(isinstance(x,ast.Name) and x.id=='client' for x in ast.walk(n))}
    if uses!=KUBE_FUNCTIONS:raise ValueError('native Kubernetes import sites changed')
    lines=source.splitlines(keepends=True)
    for n in reversed(tree.body):
        if isinstance(n,ast.FunctionDef) and n.name in uses:
            lines.insert(n.body[0].lineno-1,'    from kubernetes import client\n')
    source=''.join(lines)
    if source.count('from kubernetes import client, config\n')!=1:
        raise ValueError('native Kubernetes import declaration changed')
    result[ALIASES['utils']]=source.replace('from kubernetes import client, config\n','',1).encode()
    for relative,raw in result.items():compile(raw,relative,'exec')
    return result


@contextmanager
def native_controller(overlay_root,expected_manifest_sha):
    """Reuse checked source overlay loader, reject collisions, remove own imports.

    Caller must finish/cancel native tasks and close actual resources before
    leaving; this function does not claim to manage service lifecycle itself.
    """
    root=Path(overlay_root); raw=(root/'manifest.json').read_bytes()
    if sha256(raw).hexdigest()!=expected_manifest_sha:
        raise ValueError('controller manifest identity drift')
    if set(ALIASES)&sys.modules.keys():
        raise RuntimeError('native flat controller modules already imported')
    manifest=json.loads(raw)
    for name,relative in ALIASES.items():
        if relative not in manifest or sha256((root/relative).read_bytes()).hexdigest()!=manifest[relative]:
            raise ValueError('controller module identity drift: '+name)
    finder=SourceOverlay(root,manifest,module_aliases=ALIASES)
    sys.meta_path.insert(0,finder)
    try:
        scheduler=importlib.import_module('scheduler')
        importlib.import_module('algs')
        yield scheduler
    finally:
        sys.meta_path.remove(finder)
        for name in ALIASES:
            module=sys.modules.get(name)
            if module is not None and getattr(module,'__loader__',None) is finder:
                del sys.modules[name]
