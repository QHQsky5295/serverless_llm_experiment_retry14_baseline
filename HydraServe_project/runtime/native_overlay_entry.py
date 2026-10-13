"""Stdlib-only import overlay installer; run original app/provider as __main__.

No vLLM/torch imports here, preserving the original app's load/import overlap.
Only exact, manifest-bound module sources are replaced; package resources and
compiled extensions still come from the previously qualified installed package.
"""
from hashlib import sha256
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


class SourceOverlay(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def __init__(self,root,manifest):
        self.root,self.manifest=Path(root),manifest
        self.modules={name[:-3].replace('/','.') :name for name in manifest
                      if name.startswith('vllm/') and name.endswith('.py')}

    def find_spec(self,fullname,path=None,target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_file_location(fullname,
                self.root/self.modules[fullname],loader=self)
        return None

    def create_module(self,spec):return None

    def exec_module(self,module):
        relative=self.modules[module.__name__]
        raw=(self.root/relative).read_bytes()
        if sha256(raw).hexdigest()!=self.manifest[relative]:
            raise ValueError('installed overlay identity drift')
        exec(compile(raw,str(self.root/relative),'exec'),module.__dict__)


def main():
    root=Path(os.environ['HYDRASERVE_OVERLAY_ROOT'])
    raw=(root/'manifest.json').read_bytes()
    if sha256(raw).hexdigest()!=os.environ['HYDRASERVE_OVERLAY_SHA256']:
        raise ValueError('overlay manifest identity drift')
    manifest=json.loads(raw)
    sys.meta_path.insert(0,SourceOverlay(root,manifest))
    entry=os.environ['HYDRASERVE_NATIVE_ENTRY']
    source=(root/entry).read_bytes()
    if sha256(source).hexdigest()!=manifest[entry]:raise ValueError('native entry drift')
    exec(compile(source,str(root/entry),'exec'),{'__name__':'__main__','__file__':str(root/entry)})


if __name__=='__main__':main()
