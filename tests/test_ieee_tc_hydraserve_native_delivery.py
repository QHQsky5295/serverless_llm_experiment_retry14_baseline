"""CPU-only input/layout checks, not native model or inference qualification."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from HydraServe_project.runtime import native_delivery as delivery
from HydraServe_project.runtime.native_overlay_entry import SourceOverlay
from HydraServe_project.scripts.probe_layer_identity import NATIVE, MODEL


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_native_overlay_scope(self):
        overlays=delivery.render_overlays(NATIVE)
        provider=overlays[delivery.PROVIDER].decode()
        self.assertIn('model_list = [os.environ["HYDRASERVE_LOCAL_MODEL_ID"]]',provider)
        self.assertNotIn('from modelscope',provider)
        # The packing/partition/wire and entire main loop stay original except
        # the already-qualified bind-address adaptation.
        def bodies(source):
            return {n.name:ast.dump(n) for n in ast.parse(source).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(bodies(provider),bodies((NATIVE/delivery.PROVIDER).read_text()))
        app=overlays[delivery.APP].decode()
        self.assertEqual(app.count('enable_lora=True,'),3)
        self.assertEqual(delivery.native_lora_token_limit(NATIVE),65528)
        self.assertEqual(app.count('max_num_batched_tokens=65528,'),3)
        self.assertLess(app.index('asyncio.run(start_load_model_shm'),app.index('import vllm'))

    def test_lora_capacity_matches_original_instance_admission_all_engine_sites(self):
        old=delivery.render_overlays(NATIVE)
        new=delivery.render_overlays(NATIVE,lora_admission_capacity=True)
        batch=delivery.native_admission_batch(NATIVE)
        self.assertEqual(batch,8)  # Pinned source default, not the implementation's constant.
        tree=ast.parse(new[delivery.APP])
        sites=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id in ('EngineArgs','AsyncEngineArgs')]
        self.assertEqual(len(sites),3)
        for site in sites:
            kw={k.arg:k.value for k in site.keywords}
            self.assertNotIn('max_cpu_loras',kw)  # Native default follows max_loras exactly.
            self.assertNotIn('max_num_seqs',kw)
            expression=compile(ast.Expression(kw['max_loras']),'native-capacity','eval')
            for pp in (1,2,3,4):
                self.assertEqual(eval(expression,{'os':types.SimpleNamespace(environ={'PP_SIZE':str(pp)})}),batch*pp)
        class RemoveCapacity(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id in ('EngineArgs','AsyncEngineArgs'):
                    node.keywords=[k for k in node.keywords if k.arg!='max_loras']
                return node
        self.assertEqual(ast.dump(ast.parse(old[delivery.APP])),ast.dump(RemoveCapacity().visit(tree)))
        for name in old.keys()-{delivery.APP}:self.assertEqual(old[name],new[name],name)
        # Destination's current PP_SIZE is explicitly updated before its engine.
        app=new[delivery.APP].decode()
        self.assertLess(app.index('os.environ["PP_SIZE"] = str(dest_pp_size)'),app.index('engine_args = AsyncEngineArgs('))

    def test_lora_capacity_source_drift_rejected(self):
        original=Path.read_bytes
        def read(path):
            data=original(path)
            return data+b'\n' if path.name=='scheduler.py' else data
        with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ValueError,'admission source drift'):
            delivery.native_admission_batch(NATIVE)

    def fixture(self,missing=False):
        mapping={'a':'part1.safetensors','b':'part2.safetensors'}
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
        values={'part1.safetensors':{'a':object()},'part2.safetensors':{'b':object()}}
        if missing:values['part2.safetensors']={}
        class Handle:
            def __init__(self,path,framework):
                assert framework=='pt'
                self.values=values[Path(path).name]
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def keys(self):return self.values.keys()
            def get_tensor(self,key):return self.values[key]
        return [self.root/x for x in mapping.values()],Handle,values

    def test_two_existing_shards_identity_no_copy(self):
        files,opened,values=self.fixture()
        rows=dict(delivery.readonly_shard_iterator(reversed(files),opened))
        self.assertIs(rows['a'],values[files[0].name]['a'])
        self.assertIs(rows['b'],values[files[1].name]['b'])

    def test_missing_or_duplicate_shard_rejected(self):
        files,opened,_=self.fixture()
        for candidate in ([],files[:1],files+[files[0]]):
            with self.assertRaises(ValueError):list(delivery.readonly_shard_iterator(candidate,opened))

    def test_missing_tensor_rejected(self):
        files,opened,_=self.fixture(missing=True)
        with self.assertRaisesRegex(ValueError,'missing'):
            list(delivery.readonly_shard_iterator(files,opened))

    def test_metadata_is_small_and_reuses_real_shards(self):
        out=self.root/'views'
        p=delivery.materialize_metadata(out,'llama2_7b',MODEL,native_root=NATIVE)
        self.assertEqual(p['partitions'],{1:[32],2:[16,32],3:[11,22,32],4:[8,16,24,32]})
        real_bytes=sum(x.stat().st_size for x in out.rglob('*') if x.is_file() and not x.is_symlink())
        self.assertLess(real_bytes,1024**2)
        view=out/'llama2_7b-3-4'
        self.assertEqual(json.loads((view/'config.json').read_text())['num_hidden_layers'],8)
        self.assertTrue((view/'model-00001-of-00002.safetensors').is_symlink())
        mapping=json.loads((view/'duplicate_state_dict-1-2.json').read_text())
        self.assertEqual(mapping['model.layers.0.self_attn.qkv_proj.weight'],
                         'model.layers.8.self_attn.qkv_proj.weight')
        self.assertEqual(mapping['lm_head.weight'],'lm_head.weight')
        with self.assertRaises(FileExistsError):
            delivery.materialize_metadata(out,'llama2_7b',MODEL,native_root=NATIVE)

    def test_overlay_loader_digest_and_no_heavy_import(self):
        path=self.root/'vllm/test.py';path.parent.mkdir();path.write_text('VALUE = 42\n')
        loader=SourceOverlay(self.root,{'vllm/test.py':sha256(path.read_bytes()).hexdigest()})
        m=types.ModuleType('vllm.test');loader.exec_module(m)
        self.assertEqual(m.VALUE,42)
        self.assertIsNone(loader.find_spec('torch'))
        path.write_text('VALUE = 43\n')
        with self.assertRaisesRegex(ValueError,'drift'):loader.exec_module(m)


if __name__=='__main__':unittest.main()
