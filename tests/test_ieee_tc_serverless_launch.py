"""Native script adaptation checks; these do not qualify actual Ray workers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/prepare_ieee_tc_serverless_stack.py"
spec = importlib.util.spec_from_file_location("tc_launch", HELPER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
MAIN = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")


class NativeLaunchTests(unittest.TestCase):
    def test_existing_embedding_selector_requires_complete_current_content(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'adapter_config.json'
            config.write_text(json.dumps(dict(target_modules=['q_proj'], modules_to_save=None)))
            weights = root/'adapter_model.safetensors'
            save_file({'base_model.model.layers.0.self_attn.q_proj.lora_A.weight': torch.zeros(1, 1)}, weights)
            policy = launch.existing_pool_embedding_policy({'a': str(root)})
            self.assertTrue(policy['disable_lora_embeddings'])
            self.assertEqual(policy['inspected_adapters'], 1)
            self.assertFalse(policy['independent_numerical_correctness'])
            (root/'added_tokens.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'cannot discard'):
                launch.existing_pool_embedding_policy({'a': str(root)})
            (root/'added_tokens.json').unlink()
            save_file({'base_model.model.embed_tokens.lora_embedding_A': torch.zeros(1, 1)}, weights)
            with self.assertRaisesRegex(ValueError, 'cannot discard'):
                launch.existing_pool_embedding_policy({'a': str(root)})
            weights.unlink()
            with self.assertRaisesRegex(ValueError, 'cannot discard'):
                launch.existing_pool_embedding_policy({'a': str(root)})

    def test_model_qualification_uses_native_format_and_one_explicit_instance(self):
        config = launch.native_model_config(Path('/models/vllm/existing'), Path('/source/model'))
        self.assertEqual(config['model'], 'existing')
        self.assertNotIn('load_format', config['backend_config'])
        self.assertFalse(config['backend_config']['skip_store_model_registration'])
        self.assertNotIn('enable_lora', config['backend_config'])
        self.assertEqual(config['auto_scaling_config']['min_instances'], 1)
        self.assertEqual(config['auto_scaling_config']['max_instances'], 1)

    def test_native_response_requires_observed_count_and_instance(self):
        body = dict(id='r', usage=dict(completion_tokens=5, prompt_tokens=12),
                    metrics=dict(instance_id='native'), choices=[{}])
        launch.validate_native_response(body, 'r', 5, 12)
        for key, value in [('id', 'wrong'), ('error', 'bad'), ('metrics', {}),
                           ('usage', dict(completion_tokens=4, prompt_tokens=12))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.validate_native_response(dict(body, **{key: value}), 'r', 5, 12)

    def sources(self):
        return {name: (ROOT / "scripts" / name).read_text() for name in launch.SOURCE_SHA}

    def render(self, **changes):
        options = dict(script_dir=Path("/tmp/tc-launch-test/scripts"),
                       private_root=Path("/tmp/tc-launch-test/runtime"),
                       main_repo=MAIN, gpu_ids=(0, 1, 2, 3))
        options.update(changes)
        return launch.render(self.sources(), **options)

    def test_exact_original_sources_preserved_and_all_rendered_scripts_parse(self):
        rendered, _ = self.render()
        for name, source in rendered.items():
            with self.subTest(name=name):
                subprocess.run(["bash", "-n"], input=source, text=True, check=True)
                self.assertEqual(launch.sha((ROOT / "scripts" / name).read_bytes()), launch.SOURCE_SHA[name])

    def test_global_cleanup_source_mutation_and_log_overwrite_removed(self):
        rendered, _ = self.render()
        stack = rendered["start_serverlessllm_stack.sh"]
        for forbidden in ("stop_serverlessllm_stack.sh", "sync_serverlessllm_runtime_sources.sh",
                          "kill-session", 'rm -f "${SERVE_LOG_PATH}"', '"bash -lc'):
            self.assertNotIn(forbidden, stack)
        self.assertIn('command tmux -f "${SLLM_TC_TMUX_CONFIG}" -S "${SLLM_TC_TMUX_SOCKET}"', stack)
        self.assertLess(stack.index(" verify --manifest "), stack.index("tmux new-session"))
        self.assertIn('unset TMUX TMUX_PANE', stack)

    def test_one_worker_and_aggregate_eight_gib_not_per_node(self):
        rendered, allocation = self.render()
        self.assertEqual(sum(allocation.values()), 8 * 1024**3)
        self.assertEqual(len(allocation), 2)
        self.assertIn("export SLLM_SINGLE_HOST_MULTI_GPU=1", rendered["start_serverlessllm_stack.sh"])
        for role in ("head", "worker"):
            source = rendered[f"run_serverlessllm_{role}.sh"]
            self.assertIn(f"RAY_OBJECT_STORE_MEMORY_BYTES={4 * 1024**3}", source)
            self.assertNotIn('${SLLM_RAY_OBJECT_STORE_MEMORY_BYTES:-}', source)

    def test_native_loader_store_gpu_set_and_private_spill(self):
        rendered, _ = self.render()
        self.assertIn("--enable-storage-aware", rendered["run_serverlessllm_serve.sh"])
        self.assertIn("CUDA_VISIBLE_DEVICES=${WORKER_GPUS} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh",
                      rendered["start_serverlessllm_stack.sh"])
        for role in ("head", "worker"):
            self.assertIn('--object-spilling-directory="${SLLM_TC_SPILL}/' + role + '"',
                          rendered[f"run_serverlessllm_{role}.sh"])
        self.assertIn('--temp-dir="${SLLM_TC_RAY_TEMP}"', rendered["run_serverlessllm_head.sh"])
        self.assertNotIn('--temp-dir=', rendered["run_serverlessllm_worker.sh"])

    def test_source_drift_refused(self):
        sources = self.sources()
        sources["run_serverlessllm_worker.sh"] += "\n# changed\n"
        with self.assertRaisesRegex(ValueError, "source drift"):
            launch.render(sources, script_dir=Path('/tmp/a'), private_root=Path('/tmp/b'),
                          main_repo=MAIN, gpu_ids=(0,))

    def test_invalid_gpu_and_shell_paths_refused(self):
        for ids in ((), (0, 0), (4,), (True,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.render(gpu_ids=ids)
        for path in (Path("relative"), Path("/tmp/a b"), Path("/tmp/a;touch-oops")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.render(script_dir=path)

    def test_prepare_exclusive_and_does_not_start_services(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            manifest = launch.prepare(output, private, MAIN, (0, 1))
            self.assertFalse(manifest["performance_run_authorized"])
            self.assertFalse(manifest["actual_workers_verified"])
            self.assertFalse((private / "tmux.sock").exists())
            self.assertFalse((output / "serve.log").exists())
            self.assertEqual((output / 'tmux.conf').read_text(), launch.TMUX_CONFIG)
            with self.assertRaises(FileExistsError):
                launch.prepare(output, private, MAIN, (0, 1))

    def test_existing_results_parent_symlink_uses_one_canonical_identity(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            real = root / 'real'
            real.mkdir()
            alias = root / 'results'
            alias.symlink_to(real, target_is_directory=True)
            output, private = alias / 'view', root / 'private'
            manifest = launch.prepare(output, private, MAIN, (0,))
            self.assertEqual(manifest['script_dir'], str(real / 'view'))
            self.assertEqual(manifest['requested_script_dir'], str(output))
            env = dict(os.environ)
            env.pop('FAASLORA_TC_LAUNCH_RECEIPT', None)
            check = subprocess.run(['/usr/bin/python3', str(HELPER), 'verify', '--manifest',
                                    str(output / 'launch_manifest.json')], env=env,
                                   text=True, capture_output=True, timeout=10)
            self.assertIn('missing guarded TC launch receipt', check.stderr)
            self.assertNotIn('launcher identity differs', check.stderr)

    def test_generated_stack_refuses_before_native_launch_outside_guard(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0,))
            env = dict(os.environ)
            env.pop("FAASLORA_TC_LAUNCH_RECEIPT", None)
            result = subprocess.run(["bash", str(output / "start_serverlessllm_stack.sh")],
                                    env=env, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing guarded TC launch receipt", result.stderr)
            self.assertFalse((private / "tmux.sock").exists())
            self.assertFalse((output / "serve.log").exists())

    def test_changed_leaf_is_rejected_before_gate_or_native_launch(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0,))
            leaf = output / "run_serverlessllm_head.sh"
            leaf.write_text(leaf.read_text() + "\n# changed after preparation\n")
            with self.assertRaisesRegex(ValueError, "script changed"):
                launch.verify(output / "launch_manifest.json")

    def test_changed_aggregate_contract_is_not_accepted(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            manifest = launch.prepare(output, private, MAIN, (0,))
            manifest["object_store_bytes_total"] = 16 * 1024**3
            (output / "launch_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "aggregate object-store contract differs"):
                launch.verify(output / "launch_manifest.json")

    def test_actual_leaf_arguments_have_distinct_allocations_and_no_global_stop(self):
        with tempfile.TemporaryDirectory(prefix="tcs-test-") as directory:
            root = Path(directory)
            output, private = root / "view", root / "private"
            launch.prepare(output, private, MAIN, (0, 1))
            fake = root / "capture"
            fake.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o700)
            env = dict(os.environ, SLLM_HEAD_RAY_BIN=str(fake), SLLM_WORKER_RAY_BIN=str(fake),
                       SLLM_TC_RAY_TEMP=str(private / "ray_tmp/ray"), SLLM_TC_SPILL=str(output / "spill"),
                       SLLM_STORE_PATH=str(root / "store"), SLLM_WORKER_NUM_GPUS="2")
            total = 0
            for role in ("head", "worker"):
                result = subprocess.run(["bash", str(output / f"run_serverlessllm_{role}.sh")],
                                        env=env, text=True, capture_output=True, timeout=10, check=True)
                args = json.loads(result.stdout)
                size = next(arg for arg in args if arg.startswith("--object-store-memory="))
                total += int(size.split('=')[1])
                self.assertIn("--block", args)
                self.assertIn(f"--object-spilling-directory={output}/spill/{role}", args)
                resources = json.loads(next(a.split('=', 1)[1] for a in args if a.startswith('--resources=')))
                self.assertEqual(resources, {'control_node': 1} if role == 'head' else
                                 {'worker_node': 1, 'worker_id_0': 1})
                explicit = {'control_node': 1} if role == 'head' else {'worker_node': 1, 'worker_id_0': 1}
                override = dict(env, **{f'SLLM_{role.upper()}_RESOURCES': json.dumps(explicit)})
                result = subprocess.run(['bash', str(output / f'run_serverlessllm_{role}.sh')],
                                        env=override, text=True, capture_output=True, timeout=10, check=True)
                args = json.loads(result.stdout)
                resources = json.loads(next(a.split('=', 1)[1] for a in args if a.startswith('--resources=')))
                self.assertEqual(resources, explicit)
            self.assertEqual(total, 8 * 1024**3)

    def test_private_pane_retention_configuration_is_hash_bound(self):
        with tempfile.TemporaryDirectory(prefix='tcs-test-') as directory:
            root = Path(directory)
            output = root / 'view'
            launch.prepare(output, root / 'private', MAIN, (0,))
            (output / 'tmux.conf').write_text('set-window-option -g remain-on-exit off\n')
            with self.assertRaisesRegex(ValueError, 'tmux configuration changed'):
                launch.verify(output / 'launch_manifest.json')

    def test_ray_only_executes_same_prefix_without_store_api_or_model(self):
        full, _ = self.render()
        probe, _ = self.render(ray_only=True)
        marker = 'wait_for_workers "${EXPECTED_WORKERS}"\n'
        self.assertEqual(full['start_serverlessllm_stack.sh'].split(marker)[0],
                         probe['start_serverlessllm_stack.sh'].split(marker)[0])
        suffix = probe['start_serverlessllm_stack.sh'].split(marker)[1]
        self.assertNotIn('new-session', suffix)
        self.assertIn('ray-only infrastructure', suffix)

    def nodes(self):
        return [dict(Alive=True, NodeID='head', Resources=dict(control_node=1, object_store_memory=4 * launch.GIB)),
                dict(Alive=True, NodeID='worker', Resources=dict(worker_node=1, GPU=4, object_store_memory=4 * launch.GIB))]

    def test_live_node_capacity_and_roles_match(self):
        self.assertEqual(set(launch.validate_ray_nodes(self.nodes(), 4)), {'head', 'worker_0'})

    def test_live_node_resource_or_membership_drift_is_not_qualified(self):
        for change in ('capacity', 'gpu', 'role', 'third', 'dead'):
            with self.subTest(change=change):
                nodes = self.nodes()
                if change == 'capacity':
                    nodes[1]['Resources']['object_store_memory'] *= 2
                elif change == 'gpu':
                    nodes[1]['Resources']['GPU'] = 1
                elif change == 'role':
                    nodes[1]['Resources']['control_node'] = 1
                elif change == 'third':
                    nodes.append(dict(Alive=True, NodeID='foreign', Resources={}))
                else:
                    nodes[1]['Alive'] = False
                with self.assertRaises(ValueError):
                    launch.validate_ray_nodes(nodes, 4)


class NativeCheckpointLayoutTests(unittest.TestCase):
    def test_only_complete_unscaled_derived_rope_is_classified(self):
        names = {f'model.layers.{i}.self_attn.rotary_emb.inv_freq' for i in range(2)}
        config = dict(num_hidden_layers=2, rope_scaling=None)
        self.assertEqual(launch.derived_rope_sources(names | {'lm_head.weight'}, config), names)
        self.assertEqual(launch.derived_rope_sources({'lm_head.weight'}, config), set())
        for keys, cfg in (({next(iter(names))}, config), (names | {'wrong.rotary_emb.inv_freq'}, config),
                          (names, dict(config, rope_scaling={'type': 'linear'}))):
            with self.assertRaises(ValueError):
                launch.derived_rope_sources(keys, cfg)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'native tensor environment required')
    def test_stored_rope_requires_exact_formula_or_exact_serialization_roundtrip(self):
        import torch
        expected = 1.0 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        self.assertEqual(launch.validate_stored_rope(expected, expected), 'exact_config_formula_at_stored_dtype')
        stored = expected.half().float()
        self.assertEqual(launch.validate_stored_rope(stored, expected), 'exact_fp16_serialization_roundtrip')
        damaged = stored.clone()
        damaged[1] = torch.nextafter(damaged[1], torch.tensor(float('inf')))
        with self.assertRaises(ValueError):
            launch.validate_stored_rope(damaged, expected)
        with self.assertRaises(ValueError):
            launch.validate_stored_rope(stored[:-1], expected)

    def test_partition_reader_crosses_boundaries_and_hashes_each_part(self):
        import hashlib
        import io
        raw = launch.NativeTensorStream([io.BytesIO(b'abc'), io.BytesIO(b'defg')])
        self.assertEqual(raw.read(0), b'')
        self.assertEqual(raw.read(4), b'abcd')
        self.assertEqual(raw.tell(), 4)
        self.assertEqual(raw.read(4), b'efg')
        self.assertEqual(raw.tell(), 7)
        self.assertEqual(raw.read(1), b'')
        self.assertEqual([d.hexdigest() for d in raw.digests],
                         [hashlib.sha256(p).hexdigest() for p in (b'abc', b'defg')])
        with self.assertRaises(ValueError):
            raw.read(-1)

    def test_partition_discovery_rejects_gaps_extra_files_and_links(self):
        with tempfile.TemporaryDirectory(prefix='tcs-test-') as directory:
            rank = Path(directory)
            (rank / 'tensor_index.json').write_text('{}')
            (rank / 'tensor.data_0').write_bytes(b'abc')
            (rank / 'tensor.data_1').write_bytes(b'defg')
            self.assertEqual([p.name for p in launch.checkpoint_parts(rank)],
                             ['tensor.data_0', 'tensor.data_1'])
            (rank / 'tensor.data_1').rename(rank / 'tensor.data_2')
            with self.assertRaises(ValueError):
                launch.checkpoint_parts(rank)
            (rank / 'tensor.data_2').unlink()
            (rank / 'tensor.data_1').symlink_to(rank / 'tensor.data_0')
            with self.assertRaises(ValueError):
                launch.checkpoint_parts(rank)

    def test_projection_packing_is_complete_and_ordered(self):
        source = {'model.layers.0.self_attn.' + p + '.weight'
                  for p in ('q_proj', 'k_proj', 'v_proj')}
        source |= {'model.layers.0.mlp.' + p + '.weight' for p in ('gate_proj', 'up_proj')}
        source.add('model.norm.weight')
        source.add('lm_head.weight')
        mapping = launch.checkpoint_tensor_sources(source)
        self.assertEqual(mapping['model.layers.0.self_attn.qkv_proj.weight'],
                         ['model.layers.0.self_attn.' + p + '.weight' for p in ('q_proj', 'k_proj', 'v_proj')])
        self.assertEqual(mapping['model.layers.0.mlp.gate_up_proj.weight'],
                         ['model.layers.0.mlp.' + p + '.weight' for p in ('gate_proj', 'up_proj')])
        self.assertEqual(mapping['model.norm.weight'], ['model.norm.weight'])
        self.assertEqual(mapping['lm_head.weight'], ['lm_head.weight'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            launch.checkpoint_tensor_sources(source - {'model.layers.0.self_attn.k_proj.weight'})

    def test_export_refuses_outside_existing_guard_before_loading_or_writing(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix='tcs-test-') as directory:
            output = Path(directory) / 'export.json'
            original = os.environ.pop('FAASLORA_TC_LAUNCH_RECEIPT', None)
            try:
                with self.assertRaisesRegex(RuntimeError, 'missing guarded'):
                    launch.export_checkpoint(SimpleNamespace(main_repo=MAIN, output=output))
            finally:
                if original is not None:
                    os.environ['FAASLORA_TC_LAUNCH_RECEIPT'] = original
            self.assertFalse(output.exists())

    def test_complete_contiguous_fp16_layout(self):
        index = {'a': [0, 12, [2, 3], [3, 1], 'torch.float16'],
                 'b': [12, 6, [3], [1], 'torch.float16']}
        self.assertEqual(len(launch.validate_checkpoint_index(index, {'a': [], 'b': []}, 18)), 2)
        for field, value in ((0, 2), (1, 14), (2, [2, 4]), (3, [1, 2]), (4, 'torch.bfloat16')):
            changed = json.loads(json.dumps(index))
            changed['a'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.validate_checkpoint_index(changed, {'a': [], 'b': []}, 18)
        for size in (16, 20):
            with self.assertRaises(ValueError):
                launch.validate_checkpoint_index(index, {'a': [], 'b': []}, size)
        with self.assertRaisesRegex(ValueError, 'keys differ'):
            launch.validate_checkpoint_index(index, {'a': []}, 18)


if __name__ == "__main__":
    unittest.main()
