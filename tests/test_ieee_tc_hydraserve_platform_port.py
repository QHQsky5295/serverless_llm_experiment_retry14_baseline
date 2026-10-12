"""CPU-only platform seams; no Kubernetes, Podman, model, or GPU operation.

Synthetic lifecycle endpoints are test doubles only. Their success is not
evidence that actual worker start/stop, isolation or native inference works.
"""
import ast
import asyncio
from contextlib import redirect_stdout
from hashlib import sha256
import io
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime.platform_port import (
    SOURCE_DIR, SOURCE_SHA256, render_overlays)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "vendor_new_baselines/HydraServe_main_20260520"


def load_resource(source, **overrides):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in ("utils", "ImageInfo"))]
    namespace = {"__name__": __name__, "ImageList": {"vllm": "native-image"}}
    namespace.update(overrides)
    exec(compile(tree, "native-resource-manager", "exec"), namespace)
    return namespace


def definitions(source):
    tree = ast.parse(source)
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


class PlatformDouble:
    def __init__(self, inventory=None):
        self.nodes = inventory or {"physical-host": (4, 24, 1, 80)}
        self.created, self.deleted, self.address_checks = [], [], []
        self.confirm_delete = lambda name: True
        self.fail_create = False

    def inventory(self):
        return self.nodes

    def storage_ip(self, node):
        return "192.0.2.10"

    async def create_worker(self, name, image, env, **kwargs):
        if self.fail_create:
            raise RuntimeError("test start failure")
        self.created.append((name, image, dict(env), kwargs))

    async def worker_ip(self, name):
        self.address_checks.append(name)
        return "192.0.2." + str(20 + len(self.address_checks))

    async def delete_worker(self, name):
        self.deleted.append(name)

    async def worker_deleted(self, name):
        return self.confirm_delete(name)


class OverlayTests(unittest.TestCase):
    def test_all_input_identities_and_no_vendor_mutation(self):
        before = {name: (NATIVE / SOURCE_DIR / name).read_bytes()
                  for name in SOURCE_SHA256}
        overlays = render_overlays(NATIVE)
        self.assertEqual(set(overlays), {SOURCE_DIR + "/" + name
            for name in ("resource_manager.py", "scheduler.py")})
        for name, raw in before.items():
            self.assertEqual(sha256(raw).hexdigest(), SOURCE_SHA256[name])
            self.assertEqual(raw, (NATIVE / SOURCE_DIR / name).read_bytes())

    def test_rejects_modified_native_source(self):
        read = Path.read_bytes

        def changed(path):
            raw = read(path)
            return raw + b"\n" if path.name == "algs.py" else raw

        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(ValueError, "unqualified.*algs.py"):
                render_overlays(NATIVE)

    def test_policy_ast_unchanged(self):
        overlays = render_overlays(NATIVE)
        allowed = {"ResourceManager": {"__init__", "get_storage_server_ip",
                   "create_pods", "release_resource"}, "Scheduler": {"__init__"}}
        for file in ("resource_manager.py", "scheduler.py"):
            old = definitions((NATIVE / SOURCE_DIR / file).read_bytes())
            new = definitions(overlays[SOURCE_DIR + "/" + file])
            self.assertEqual(set(old), set(new))
            for name in old:
                if name not in allowed:
                    self.assertEqual(dump(old[name]), dump(new[name]), name)
                    continue
                old_methods = {m.name: m for m in old[name].body
                               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                new_methods = {m.name: m for m in new[name].body
                               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                self.assertEqual(set(old_methods), set(new_methods))
                for method in old_methods.keys() - allowed[name]:
                    self.assertEqual(dump(old_methods[method]), dump(new_methods[method]),
                                     name + "." + method)
        old_init = next(m for m in old["Scheduler"].body if getattr(m, "name", "") == "__init__")
        new_init = next(m for m in new["Scheduler"].body if getattr(m, "name", "") == "__init__")
        # Only the keyword injection and ResourceManager constructor call differ.
        self.assertEqual([dump(n) for n in old_init.body[1:]],
                         [dump(n) for n in new_init.body[1:]])


class PlatformTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SLOW_EXPR": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.text = render_overlays(NATIVE)[SOURCE_DIR + "/resource_manager.py"]
        self.ns = load_resource(self.text)
        self.platform = PlatformDouble()
        self.manager = self.make_manager(self.platform)

    def make_manager(self, platform):
        with redirect_stdout(io.StringIO()):
            return self.ns["ResourceManager"](None, None, platform=platform)

    def allocation(self):
        alloc = self.ns["AllocatedResource"](1, [0], [19], [1])
        return alloc, self.manager.claim_allocated_resource(alloc)

    async def test_real_node_shape_and_native_best_fit(self):
        self.assertEqual(self.manager.node_list, ["physical-host"])
        self.assertEqual(self.manager.rgpus, [[24, 24, 24, 24]])
        selections = [self.allocation()[1][0] for _ in range(4)]
        self.assertEqual([u.gpu_id for u in selections], [0, 1, 2, 3])
        # Native <=6 GB remainder rule is preserved, not a new quota policy.
        self.assertEqual([u.num_gpu_mem for u in selections], [24] * 4)
        self.assertEqual([u.alloc_gpu_mem for u in selections], [19] * 4)
        self.assertEqual(self.manager.claim_resource(0, 19), -1)

    async def test_worker_env_and_gpu_identity_forwarding(self):
        alloc, usages = self.allocation()
        ips = await self.manager.create_pods("llama2_7b/0", alloc, usages)
        name, image, env, options = self.platform.created[0]
        self.assertEqual(name, usages[0].pod_name)
        self.assertEqual(image, "native-image")
        self.assertEqual(options["gpu_id"], usages[0].gpu_id)
        self.assertEqual(options["node"], "physical-host")
        self.assertEqual(options["gpu"], (24, 100))
        self.assertEqual(env, {"MODEL_ID": "llama2_7b", "REMOTE_MODEL_ID": "llama2_7b/0",
            "MODELSCOPE_CACHE": "/mnt/model-cache", "PP_SIZE": "1", "ENABLE_PARA": "0",
            "LOCAL_SERVER": "1", "PYTHONUNBUFFERED": "1", "MAX_MODEL_LEN": "3072",
            "PP_RANK": "0", "TOTAL_GPU": "24", "STORAGE_IP": "192.0.2.10",
            "DEST_PP_SIZE": "1", "BATCH_SIZE": "8"})
        self.assertEqual(ips, ["192.0.2.21"])

    async def test_native_pipeline_env_is_not_flattened(self):
        platform = PlatformDouble({"physical-a": (1, 24, 1, 80),
                                   "physical-b": (1, 24, 1, 80)})
        manager = self.make_manager(platform)
        alloc = self.ns["AllocatedResource"](2, [0, 1], [19, 10], [1, 1], True)
        uses = manager.claim_allocated_resource(alloc)
        await manager.create_pods("llama2_7b/0", alloc, uses)
        self.assertEqual([x[2]["PP_SIZE"] for x in platform.created], ["2", "2"])
        self.assertEqual([x[2]["PP_RANK"] for x in platform.created], ["0", "1"])
        self.assertEqual([x[2]["DEST_PP_SIZE"] for x in platform.created], ["1", "2"])

    async def test_no_resource_credit_before_confirmed_exit(self):
        _, uses = self.allocation()
        observed = []

        def confirm(name):
            observed.append(list(self.manager.rgpus[0]))
            return len(observed) == 2

        self.platform.confirm_delete = confirm
        await self.manager.release_resource(uses)
        self.assertEqual(observed, [[0, 24, 24, 24]] * 2)
        self.assertEqual(self.manager.rgpus, [[24] * 4])
        self.assertEqual(self.manager.pods, [[]])

    async def test_release_error_does_not_forge_resource_release(self):
        _, uses = self.allocation()

        def fail(name):
            raise RuntimeError("test ownership check failed")

        self.platform.confirm_delete = fail
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            await self.manager.release_resource(uses)
        self.assertEqual(self.manager.rgpus[0][0], 0)
        self.assertEqual(self.manager.pods, [[uses[0].pod_name]])

    async def test_create_failure_propagates_and_does_not_invent_address(self):
        alloc, uses = self.allocation()
        self.platform.fail_create = True
        with self.assertRaisesRegex(RuntimeError, "start failure"):
            await self.manager.create_pods("llama2_7b/0", alloc, uses)
        self.assertEqual(self.platform.address_checks, [])

    async def test_init_still_awaits_native_worker_acknowledgement(self):
        gate, entered = asyncio.Event(), asyncio.Event()
        calls = []

        async def post(url, payload):
            calls.append((url, payload))
            entered.set()
            await gate.wait()

        self.ns["post_request_util_succ_async"] = post
        with redirect_stdout(io.StringIO()):
            task = asyncio.create_task(self.manager.initialize_pods(["192.0.2.21"]))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertFalse(task.done())
            gate.set()
            await task
        self.assertEqual(calls, [("http://192.0.2.21:8080/init",
                                  {"ip_list": ["192.0.2.21"]})])

    async def test_default_kubernetes_path_remains_compatible(self):
        nodes = {"physical-host": (4, 24, 1, 80)}
        storage = NS(metadata=NS(labels={"app": "storage-server-local"}),
                     spec=NS(node_name="physical-host"), status=NS(pod_ip="192.0.2.10"))
        pods, calls = [storage], []
        core = NS(list_namespaced_pod=lambda **kw: NS(items=pods))

        def create(api, name, image, env, **kw):
            calls.append((name, image, dict(env), kw))
            pods.append(NS(metadata=NS(labels={"app": name}), status=NS(pod_ip="192.0.2.21")))

        self.ns.update(get_node_list_with_resource=lambda api: nodes,
                       create_deployment=create, delete_deployment=lambda api, name: None,
                       check_deployment_fin=lambda api, name: True)
        with redirect_stdout(io.StringIO()):
            old = self.ns["ResourceManager"](core, object())
        alloc = self.ns["AllocatedResource"](1, [0], [19], [1])
        uses = old.claim_allocated_resource(alloc)
        self.assertEqual(await old.create_pods("llama2_7b/0", alloc, uses), ["192.0.2.21"])
        new_alloc, new_uses = self.allocation()
        await self.manager.create_pods("llama2_7b/0", new_alloc, new_uses)
        self.assertEqual(calls[0][:3], self.platform.created[0][:3])
        kwargs = dict(self.platform.created[0][3])
        kwargs.pop("gpu_id")
        self.assertEqual(calls[0][3], kwargs)
        await old.release_resource(uses)
        self.assertEqual(old.rgpus, [[24] * 4])

    async def test_unchanged_algorithm_cannot_invent_four_physical_nodes(self):
        source = (NATIVE / SOURCE_DIR / "algs.py").read_bytes()
        self.assertEqual(sha256(source).hexdigest(), SOURCE_SHA256["algs.py"])
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
                     and node.module in ("resource_manager", "scheduler"))]
        namespace = dict(AllocatedResource=self.ns["AllocatedResource"], ModelStats=object,
                         KVCacheRequiredSize=5)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_PP_SIZE", None)
            exec(compile(tree, "unchanged-native-algorithm", "exec"), namespace)
        stats = NS(model_size=14, num_profile_points=0, tpot=0, tpot_slo=100, prefill_time=0,
                   ttft_slo=5000, cost_slo=1, get_cur_cost=lambda: 0, num_output_tokens=0)
        allocations = []
        for _ in range(4):
            alloc = namespace["ours"]([max(g) for g in self.manager.get_rgpus()],
                                       [1], stats, 4, 0)
            allocations.append(alloc)
            self.manager.claim_allocated_resource(alloc)
        self.assertEqual([a.num_nodes for a in allocations], [1] * 4)
        self.assertEqual([a.nodes for a in allocations], [[0]] * 4)
        self.assertFalse(any(a.will_scale_down for a in allocations))


if __name__ == "__main__":
    unittest.main()
