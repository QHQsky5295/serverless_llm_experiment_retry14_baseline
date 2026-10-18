"""D687: render descriptions only at publication, never cache physical facts."""
import ast
import builtins
import hashlib
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from faaslora.memory import gpu_monitor as monitor
from tests.test_ieee_tc_worker_observation import NativeHostFootprint


def parent_inventory():
    """Exact frozen parent function; dependencies remain the same live module."""
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','show',
        '5bb08ffb1af21acd50e2fea728335fcae6587a99:faaslora/memory/gpu_monitor.py'],cwd=root)
    if hashlib.sha256(source).hexdigest()!='213cc94b070a1b3e0b0b1da9decdc5d0ffe0724c36aa57a892619a5c6d6eeb32':
        raise RuntimeError('D687 reference changed')
    nodes=[node for node in ast.parse(source).body
           if isinstance(node,ast.FunctionDef) and node.name=='_ieee_lora_host_inventory']
    if len(nodes)!=1:
        raise RuntimeError('D687 reference function missing')
    namespace=dict(monitor.__dict__)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<D687-exact-parent>','exec'),namespace)
    return namespace['_ieee_lora_host_inventory']


class HostMetadataProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=staticmethod(parent_inventory())

    def test_exact_parent_graph_full_projected_requested_staged(self):
        for case in ('base','packed','mixed_dtype','shared','empty'):
            native,models=NativeHostFootprint().native_models()
            if case=='packed':
                layer=models[8].loras['shared']
                models[8].loras={'qkv':SimpleNamespace(lora_a=[layer.lora_a,None],lora_b=[layer.lora_b,None])}
            elif case=='mixed_dtype':
                models[7].loras['private'].lora_a=torch.empty((4,16),dtype=torch.float32)
            elif case=='shared':
                models[7].loras['private']=models[8].loras['shared']
            elif case=='empty':
                models.clear()
            for options in ({},{'include_tensor_views':False},
                    {'include_tensor_views':False,'requested_adapter_ids':[7]}):
                with self.subTest(case=case,options=options):
                    self.assertEqual(self.parent(native,**options),monitor._ieee_lora_host_inventory(native,**options))
            if models:
                staged={9:SimpleNamespace(id=9,rank=4,loras=models[8].loras)}
                self.assertEqual(self.parent(native,staged_models=staged,include_tensor_views=False),
                    monitor._ieee_lora_host_inventory(native,staged_models=staged,include_tensor_views=False))

    def test_runtime_reads_device_once_and_no_shape_description(self):
        native,_=NativeHostFootprint().native_models()
        getter=torch.Tensor.device
        observed=[]
        def device(tensor):
            observed.append(id(tensor))
            return getter.__get__(tensor)
        with patch.object(torch.Tensor,'device',property(device)), \
             patch.object(torch.Tensor,'shape',property(lambda tensor: (_ for _ in ()).throw(AssertionError('unused shape')))):
            result=monitor._ieee_lora_host_inventory(native,include_tensor_views=False)
        self.assertEqual(result['host_tensor_storage_bytes'],768)
        self.assertEqual(len(observed),6)

    def test_dtype_rendering_is_per_distinct_adapter_dtype(self):
        native,_=NativeHostFootprint().native_models()
        observed=[]
        class Render(type):
            def __instancecheck__(cls,value):return isinstance(value,builtins.str)
            def __call__(cls,value):
                if isinstance(value,torch.dtype):observed.append(value)
                return builtins.str(value)
        class ObservedString(metaclass=Render):pass
        with patch.object(monitor,'str',ObservedString,create=True):
            result=monitor._ieee_lora_host_inventory(native,include_tensor_views=False)
        self.assertEqual(observed,[torch.float16,torch.float16])
        self.assertEqual([row['dtypes'] for row in result['host_adapter_footprints']],
                         [['torch.float16'],['torch.float16']])

    def test_invalid_tensor_errors_retain_exact_module_location(self):
        for tensor in (object(),torch.empty((0,2)),torch.empty((2,2,2)),
                       torch.empty((2,2),device='meta'),torch.empty((2,2)).to_sparse()):
            native,models=NativeHostFootprint().native_models()
            models[8].loras={'specific_module':SimpleNamespace(lora_a=tensor,lora_b=torch.empty((2,2)))}
            errors=[]
            for function in (self.parent,monitor._ieee_lora_host_inventory):
                with self.assertRaises(ValueError) as caught:function(native,include_tensor_views=False)
                errors.append(str(caught.exception))
            self.assertEqual(errors[0],errors[1])
            self.assertIn('specific_module.lora_a[0]',errors[0])


if __name__=='__main__':unittest.main()
