"""CPU import/source checks, not actual inference or fake cluster qualification."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from HydraServe_project.runtime.controller_runtime import (
    ALIASES,KUBE_FUNCTIONS,render_controller_sources,native_controller)
from HydraServe_project.runtime.request_contract import render_overlays
from HydraServe_project.runtime.request_contract import matches_worker_process
from HydraServe_project.scripts.probe_layer_identity import NATIVE


class ControllerTests(unittest.TestCase):
    def test_container_and_host_pid_are_distinct_but_birth_matches(self):
        self.assertTrue(matches_worker_process(dict(pid=1,start_ticks=123,boot_id='boot'),
            dict(pid=10001,start_ticks=123),dict(pid=1,start_ticks=123),'boot'))

    def test_worker_pid_birth_boot_mismatch_is_rejected(self):
        owner=dict(pid=10001,start_ticks=123);entry=dict(pid=1,start_ticks=123)
        for changed in (dict(pid=2),dict(start_ticks=124),dict(boot_id='other')):
            process=dict(pid=1,start_ticks=123,boot_id='boot');process.update(changed)
            self.assertFalse(matches_worker_process(process,owner,entry,'boot'))

    def test_only_platform_sdk_imports_moved(self):
        sources=render_controller_sources(NATIVE)
        original=ast.parse((NATIVE/ALIASES['utils']).read_bytes())
        actual=ast.parse(sources[ALIASES['utils']])
        original.body=[n for n in original.body if not (isinstance(n,ast.ImportFrom) and n.module=='kubernetes')]
        for n in actual.body:
            if isinstance(n,ast.FunctionDef) and n.name in KUBE_FUNCTIONS:
                self.assertEqual(ast.dump(n.body.pop(0)),ast.dump(ast.parse('from kubernetes import client').body[0]))
        self.assertEqual(ast.dump(original),ast.dump(actual))
        overlays=render_overlays(NATIVE)
        for name,path in ALIASES.items():
            if name!='utils':self.assertEqual(sources[path],overlays.get(path,(NATIVE/path).read_bytes()))

    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        sources=render_controller_sources(NATIVE)
        for name,data in sources.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        raw=json.dumps({name:sha256(data).hexdigest() for name,data in sources.items()}).encode()
        (root/'manifest.json').write_bytes(raw)
        return root,sha256(raw).hexdigest()

    def test_full_controller_import_no_kubernetes_and_original_policy(self):
        root,digest=self.fixture()
        # Only external HTTP libraries are test doubles; runtime does not stub them.
        openai=types.ModuleType('openai');openai.OpenAI=object;openai.AsyncOpenAI=object
        with patch.dict(sys.modules,{'requests':types.ModuleType('requests'),
            'aiohttp':types.ModuleType('aiohttp'),'openai':openai,'kubernetes':None}):
            with native_controller(root,digest) as module:
                class Inventory:
                    def inventory(self):return {'physical-host':(4,24,1,80)}
                    def storage_ip(self,node):return '127.0.0.1'
                with patch.dict('os.environ',{'USE_CACHE':'0','SLOW_EXPR':'0','MAX_PP_SIZE':'-1'}):
                    # The real config omits MAX_PP_SIZE; remove test-parent override.
                    import os
                    del os.environ['MAX_PP_SIZE']
                    scheduler=module.Scheduler(None,None,{'test':{'existing':(13,9000,200,1000000,1)}},platform=Inventory())
                self.assertIs(scheduler.alg,sys.modules['algs'].ours)
                self.assertEqual(scheduler.resource_manager.rgpus,[[24]*4])
                self.assertEqual(scheduler.model_list,['existing/0'])
                self.assertEqual(scheduler.models[0].batch_size,8)
                self.assertEqual(scheduler.tasks,set())
                with self.assertRaises(ModuleNotFoundError):
                    sys.modules['utils'].create_deployment(None,'unused','unused',{})
            self.assertFalse(set(ALIASES)&sys.modules.keys())

    def test_manifest_and_flat_name_collision_fail_closed(self):
        root,digest=self.fixture()
        with self.assertRaisesRegex(ValueError,'manifest'):
            with native_controller(root,'wrong'):pass
        with patch.dict(sys.modules,{'utils':types.ModuleType('utils')}):
            with self.assertRaisesRegex(RuntimeError,'already imported'):
                with native_controller(root,digest):pass
        (root/ALIASES['scheduler']).write_text('not qualified\n')
        with self.assertRaisesRegex(ValueError,'identity'):
            with native_controller(root,digest):pass


if __name__=='__main__':unittest.main()
