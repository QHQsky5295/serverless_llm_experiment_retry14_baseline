"""Focused, GPU-free regression tests for the EuroSys'27 V2 runner changes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import yaml

from faaslora.datasets.workload_generator import (
    WorkloadConfig,
    WorkloadGenerator,
    _hotset_rotation_stride,
)
from faaslora.experiment.experiment_stack import ExperimentStack
from faaslora.experiment.instance_pool import InstanceSlot, Router
from faaslora.registry.schema import StorageTier
from scripts.run_all_experiments import (
    AggregateBandwidthLimiter,
    InferenceEngine as ScriptInferenceEngine,
    RequestExecutionPlan,
    RequestResult,
    ScenarioResult,
    ScenarioRunner,
    _apply_adapter_storage_env_overrides,
    _apply_explicit_env_overrides,
    _generation_contract_request_map_sha256,
    _non_feature_frozen_config_payload,
    _non_feature_frozen_config_sha256,
    _path_size_bytes,
    _validate_formal_run_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"


class RevisionV2ScenarioTests(unittest.TestCase):
    def test_ablation_runner_pins_generation_and_workload_seed(self) -> None:
        runner = (PROJECT_ROOT / "scripts" / "run_faaslora_paper_ablation_round.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'export FAASLORA_WORKLOAD_SEED="${SAMPLING_SEED}"', runner
        )
        self.assertIn(
            'export FAASLORA_GENERATION_SEED="${SAMPLING_SEED}"', runner
        )

    def test_formal_provenance_accepts_validation_and_heldout_only(self) -> None:
        digest = "a" * 64
        for role in ("validation", "heldout"):
            _validate_formal_run_provenance(
                formal_run=True,
                trace_role=role,
                system_resolved_config_sha256=digest,
            )

        with self.assertRaisesRegex(ValueError, "resolved-config SHA"):
            _validate_formal_run_provenance(
                formal_run=True,
                trace_role="validation",
                system_resolved_config_sha256="",
            )
        for role in ("smoke", "exploratory", "legacy"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "validation or heldout"
            ):
                _validate_formal_run_provenance(
                    formal_run=True,
                    trace_role=role,
                    system_resolved_config_sha256=digest,
                )

        for role in ("validation", "smoke"):
            _validate_formal_run_provenance(
                formal_run=False,
                trace_role=role,
                system_resolved_config_sha256="",
            )

    def test_revision_v2_scenarios_are_strictly_cumulative(self) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        scenarios = config["revision_v2_scenarios"]
        expected_names = [
            "v2_elastic_only",
            "v2_hit_aware_preparation",
            "v2_hierarchical_no_coord",
            "v2_full",
        ]
        self.assertEqual([scenario["name"] for scenario in scenarios], expected_names)

        expected_conceptual_features = [
            set(),
            {"hit_aware_preparation"},
            {"hit_aware_preparation", "hierarchical_residency"},
            {
                "hit_aware_preparation",
                "hierarchical_residency",
                "coordinated_admission",
            },
        ]

        observed_features = []
        for scenario in scenarios:
            self.assertEqual(scenario["baseline_type"], "faaslora_full")
            preloading = scenario["preloading"]
            coordination = scenario["resource_coordination"]

            features = set()
            preparation_enabled = (
                preloading["enabled"]
                and coordination["routing_policy"] == "adapter_affinity"
                and coordination["scale_up_handoff_enabled"]
            )
            if preparation_enabled:
                features.add("hit_aware_preparation")
            else:
                self.assertFalse(preloading["enabled"])
                self.assertEqual(coordination["routing_policy"], "least_connections")
                self.assertFalse(coordination["scale_up_handoff_enabled"])
            if preloading["hierarchical_residency_enabled"]:
                features.add("hierarchical_residency")
            if coordination["effective_capacity_admission_enabled"]:
                features.add("coordinated_admission")
            observed_features.append(features)

            hierarchy_enabled = "hierarchical_residency" in features
            self.assertEqual(preloading["dynamic_forwarding_enabled"], hierarchy_enabled)
            self.assertEqual(preloading["gpu_dynamic_forwarding_enabled"], hierarchy_enabled)
            self.assertEqual(
                preloading["host_promotion_on_nvme_hit_enabled"],
                hierarchy_enabled,
            )

            admission_enabled = "coordinated_admission" in features
            self.assertEqual(coordination["coordination_enabled"], admission_enabled)

        self.assertEqual(observed_features, expected_conceptual_features)
        for previous, current in zip(observed_features, observed_features[1:]):
            self.assertTrue(previous < current)
            self.assertEqual(len(current - previous), 1)

    def test_non_feature_digest_is_stable_across_ablation_rows_and_axes(self) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        common = {
            "experiment_cfg": {
                "num_runs": 1,
                "confidence_level": 0.95,
                "output_dir": "results/seed43",
                "results_file": "seed43.json",
            },
            "model_cfg": {
                "name": "meta-llama/Llama-2-7b-hf",
                "backend": "vllm",
                "runtime_concurrency_cap": 8,
                "max_loras": 64,
            },
            "adapters_cfg": {
                "preparation_mode": "local_frozen",
                "selected_num_adapters": 500,
                "_selected_adapter_count": 500,
                "_shared_adapter_subset_path": "/trace/seed43-subset.json",
                "adapters": [{"id": "seed43-a"}],
            },
            "storage_cfg": {
                "remote_dir": "artifacts/remote",
                "host_cache_dir": "/dev/shm/seed43",
                "nvme_cache_dir": "/tmp/seed43",
                "bandwidth_mbps": 250.0,
                "require_memory_backed_host_cache": True,
            },
            "hardware_cfg": {"gpu_memory_mb": 24576},
            "cost_model_cfg": {"serverless_idle_gpu_cost_factor": 0.238095},
            "base_coordination_cfg": {
                "min_instances": 1,
                "max_instances": 4,
                "scale_eval_interval_s": 15.0,
            },
            "full_scenario_cfg": config["revision_v2_scenarios"][-1],
        }
        digests = {
            _non_feature_frozen_config_sha256(**common, scenario_cfg=scenario)
            for scenario in config["revision_v2_scenarios"]
        }
        self.assertEqual(len(digests), 1)

        axis_changed = json.loads(json.dumps(common))
        axis_changed["experiment_cfg"].update(
            {"output_dir": "results/seed45", "results_file": "seed45.json"}
        )
        axis_changed["adapters_cfg"].update(
            {
                "selected_num_adapters": 100,
                "_selected_adapter_count": 100,
                "_shared_adapter_subset_path": "/trace/seed45-subset.json",
                "adapters": [{"id": "seed45-b"}],
            }
        )
        axis_changed["storage_cfg"].update(
            {
                "host_cache_dir": "/dev/shm/seed45",
                "nvme_cache_dir": "/tmp/seed45",
                "bandwidth_mbps": 11.9209,
            }
        )
        reference_scenario = config["revision_v2_scenarios"][-1]
        self.assertEqual(
            next(iter(digests)),
            _non_feature_frozen_config_sha256(
                **axis_changed, scenario_cfg=reference_scenario
            ),
        )

        payload = _non_feature_frozen_config_payload(
            **common, scenario_cfg=reference_scenario
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.assertNotIn("seed43-a", canonical)
        self.assertNotIn("bandwidth_mbps", canonical)
        self.assertNotIn("routing_policy", canonical)
        self.assertIn("runtime_concurrency_cap", canonical)
        self.assertIn("serverless_idle_gpu_cost_factor", canonical)

    def test_non_feature_digest_detects_runtime_autoscaler_billing_and_capacity_drift(self) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle)["revision_v2_scenarios"][-1]
        common = {
            "experiment_cfg": {"num_runs": 1},
            "model_cfg": {"backend": "vllm", "runtime_concurrency_cap": 8},
            "adapters_cfg": {"preparation_mode": "local_frozen"},
            "storage_cfg": {"remote_dir": "artifacts/remote"},
            "hardware_cfg": {"gpu_memory_mb": 24576},
            "cost_model_cfg": {"serverless_idle_gpu_cost_factor": 0.238095},
            "base_coordination_cfg": {"min_instances": 1, "max_instances": 4},
            "full_scenario_cfg": scenario,
            "scenario_cfg": scenario,
        }
        reference = _non_feature_frozen_config_sha256(**common)
        for section, key, value in (
            ("model_cfg", "runtime_concurrency_cap", 9),
            ("base_coordination_cfg", "max_instances", 3),
            ("cost_model_cfg", "serverless_idle_gpu_cost_factor", 0.5),
            ("hardware_cfg", "gpu_memory_mb", 24000),
        ):
            changed = json.loads(json.dumps(common))
            changed[section][key] = value
            self.assertNotEqual(
                reference,
                _non_feature_frozen_config_sha256(**changed),
                msg=f"failed to detect drift in {section}.{key}",
            )

        capacity_changed = json.loads(json.dumps(common))
        capacity_changed["scenario_cfg"]["preloading"]["host_capacity_mb"] = 2048
        self.assertNotEqual(
            reference, _non_feature_frozen_config_sha256(**capacity_changed)
        )

        tuning_changed = json.loads(json.dumps(common))
        tuning_changed["scenario_cfg"]["preloading"]["min_hotness"] = 0.4
        self.assertNotEqual(
            reference, _non_feature_frozen_config_sha256(**tuning_changed)
        )

    def test_router_policy_separates_load_only_from_readiness_and_handoff(self) -> None:
        coordinator = SimpleNamespace(
            compute_faaslora_host_load_ms=lambda _size_mb: 50.0,
            compute_faaslora_nvme_load_ms=lambda _size_mb: 500.0,
        )
        idle_remote = InstanceSlot("idle_remote", None, coordinator)
        busy_gpu = InstanceSlot("busy_gpu", None, coordinator)
        idle_remote.mark_adapter_tier("adapter_a", "remote")
        busy_gpu.mark_adapter_tier("adapter_a", "gpu")
        busy_gpu.active_requests = 1
        busy_gpu.scaleup_handoff_planned_adapters = ["adapter_a"]
        busy_gpu.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        busy_gpu.scaleup_handoff_request_budget = 2
        busy_gpu.scaleup_handoff_assigned_requests = 0
        pool = SimpleNamespace(get_slots=lambda: [busy_gpu, idle_remote])

        load_only = Router(pool, policy="least_connections", runtime_concurrency_cap=2)
        self.assertIs(
            load_only.select_instance("adapter_a", adapter_size_mb=30.0),
            idle_remote,
        )
        self.assertEqual(busy_gpu.scaleup_handoff_assigned_requests, 0)

        readiness_aware = Router(
            pool,
            policy="adapter_affinity",
            runtime_concurrency_cap=2,
        )
        self.assertIs(
            readiness_aware.select_instance("adapter_a", adapter_size_mb=30.0),
            busy_gpu,
        )
        self.assertEqual(busy_gpu.scaleup_handoff_assigned_requests, 1)


class RevisionV2BandwidthAndOverrideTests(unittest.TestCase):
    def test_aggregate_bandwidth_limiter_shares_one_link_across_four_transfers(self) -> None:
        async def exercise():
            limiter = AggregateBandwidthLimiter(10.0)
            started_at = time.perf_counter()
            waits_ms = await asyncio.gather(
                *(limiter.throttle(256 * 1024) for _ in range(4))
            )
            elapsed_s = time.perf_counter() - started_at
            return limiter.snapshot(), waits_ms, elapsed_s

        snapshot, waits_ms, elapsed_s = asyncio.run(exercise())

        self.assertEqual(snapshot["transfer_count"], 4)
        self.assertEqual(snapshot["total_bytes"], 1024 * 1024)
        self.assertAlmostEqual(snapshot["reservation_span_s"], 0.1, places=5)
        self.assertAlmostEqual(snapshot["achieved_reserved_mib_s"], 10.0, places=5)
        self.assertGreaterEqual(elapsed_s, 0.09)
        self.assertGreater(waits_ms[-1], waits_ms[0])

    def test_zero_bandwidth_limit_is_explicit_no_delay(self) -> None:
        async def exercise():
            limiter = AggregateBandwidthLimiter(0.0)
            started_at = time.perf_counter()
            waits_ms = await asyncio.gather(
                *(limiter.throttle(1024 * 1024 * 1024) for _ in range(4))
            )
            return limiter.snapshot(), waits_ms, time.perf_counter() - started_at

        snapshot, waits_ms, elapsed_s = asyncio.run(exercise())

        self.assertEqual(waits_ms, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(snapshot["configured_mib_s"], 0.0)
        self.assertEqual(snapshot["limit_mode"], "local_sim_no_delay")
        self.assertEqual(snapshot["transfer_count"], 4)
        self.assertEqual(snapshot["total_bytes"], 4 * 1024 * 1024 * 1024)
        self.assertEqual(snapshot["total_injected_wait_s"], 0.0)
        self.assertLess(elapsed_s, 0.05)

    def test_storage_env_overrides_prefer_explicit_mib_per_second(self) -> None:
        env = {
            "FAASLORA_REMOTE_DIR": "/tmp/v2-remote",
            "FAASLORA_HOST_CACHE_DIR": "/tmp/v2-host",
            "FAASLORA_NVME_CACHE_DIR": "/tmp/v2-nvme",
            "FAASLORA_REQUIRE_MEMORY_BACKED_HOST_CACHE": "false",
            "FAASLORA_ARTIFACT_POOL_PROFILE": "v2_pool",
            "FAASLORA_ARTIFACT_POOL_SEED": "43",
            "FAASLORA_LORA_PREPARATION_MODE": "local_frozen",
            "FAASLORA_STORAGE_BANDWIDTH_MIB_S": "119.2093",
            "FAASLORA_STORAGE_BANDWIDTH_MBPS": "999",
        }
        with patch.dict(os.environ, env, clear=True):
            adapters, storage, applied = _apply_adapter_storage_env_overrides(
                {},
                {"bandwidth_mbps": 250.0},
            )

        self.assertEqual(storage["remote_dir"], "/tmp/v2-remote")
        self.assertEqual(storage["host_cache_dir"], "/tmp/v2-host")
        self.assertEqual(storage["nvme_cache_dir"], "/tmp/v2-nvme")
        self.assertFalse(storage["require_memory_backed_host_cache"])
        self.assertAlmostEqual(storage["bandwidth_mbps"], 119.2093)
        self.assertEqual(adapters["artifact_pool_profile"], "v2_pool")
        self.assertEqual(adapters["artifact_pool_seed"], 43)
        self.assertEqual(adapters["preparation_mode"], "local_frozen")
        self.assertAlmostEqual(
            applied["FAASLORA_STORAGE_BANDWIDTH_MIB_S"],
            119.2093,
        )

    def test_rotation_helper_does_not_detach_generator_methods(self) -> None:
        cfg = WorkloadConfig(
            total_requests=12,
            active_adapter_cap=2,
            hotset_rotation_requests=3,
            hotset_rotation_mode="abrupt",
            enable_hotness_evolution=True,
            epoch_requests=4,
            enable_burst=False,
            use_azure_trace_tokens=False,
        )
        generator = WorkloadGenerator(
            ["adapter_0", "adapter_1", "adapter_2", "adapter_3"],
            cfg,
            seed=43,
        )
        generator._azure_records = []
        traces = generator.generate()

        self.assertEqual(len(traces), 12)
        self.assertTrue(all(trace.adapter_id for trace in traces))
        self.assertTrue(callable(getattr(generator, "_rotate_hotness", None)))
        self.assertTrue(callable(getattr(generator, "_sample_request", None)))

    def test_workload_and_generation_contract_env_overrides_are_typed(self) -> None:
        env = {
            "FAASLORA_WORKLOAD_SEED": "45",
            "FAASLORA_ZIPF_EXPONENT": "1.4",
            "FAASLORA_ACTIVE_ADAPTER_CAP": "500",
            "FAASLORA_HOTSET_ROTATION_REQUESTS": "100",
            "FAASLORA_HOTSET_ROTATION_MODE": "GRADUAL",
            "FAASLORA_HOTSET_OVERLAP_FRACTION": "0.5",
            "FAASLORA_GENERATION_CONTRACT": "FIXED_LENGTH_GREEDY_V1",
            "FAASLORA_FIXED_OUTPUT_MAX_TOKENS": "256",
            "FAASLORA_FIXED_PROMPT_MAX_TOKENS": "759",
        }
        with patch.dict(os.environ, env, clear=True):
            model, workload, coordination, hardware, applied = (
                _apply_explicit_env_overrides({}, {}, {}, {})
            )

        self.assertEqual(model, {})
        self.assertEqual(coordination, {})
        self.assertEqual(hardware, {})
        self.assertEqual(workload["workload_seed"], 45)
        self.assertEqual(workload["zipf_exponent"], 1.4)
        self.assertEqual(workload["active_adapter_cap"], 500)
        self.assertEqual(workload["hotset_rotation_requests"], 100)
        self.assertEqual(workload["hotset_rotation_mode"], "gradual")
        self.assertEqual(workload["hotset_overlap_fraction"], 0.5)
        self.assertEqual(workload["generation_contract"], "fixed_length_greedy_v1")
        self.assertEqual(workload["fixed_output_max_tokens"], 256)
        self.assertEqual(workload["fixed_prompt_max_tokens"], 759)
        self.assertEqual(set(applied), set(env))


class RevisionV2WorkloadAndContractTests(unittest.TestCase):
    def test_stationary_abrupt_and_gradual_rotation_strides(self) -> None:
        self.assertEqual(
            _hotset_rotation_stride(
                48,
                rotation_mode="stationary",
                overlap_fraction=0.5,
            ),
            0,
        )
        self.assertEqual(
            _hotset_rotation_stride(
                48,
                rotation_mode="abrupt",
                overlap_fraction=0.99,
            ),
            48,
        )
        self.assertEqual(
            _hotset_rotation_stride(
                48,
                rotation_mode="gradual",
                overlap_fraction=0.5,
            ),
            24,
        )
        self.assertEqual(
            _hotset_rotation_stride(
                48,
                rotation_mode="gradual",
                overlap_fraction=1.0,
            ),
            1,
        )

    def test_fixed_length_greedy_contract_builds_vllm_sampling_params(self) -> None:
        class FakeVllmEngine:
            def __init__(self) -> None:
                self.sampling_params = None

            async def generate(self, **kwargs):
                self.sampling_params = kwargs["sampling_params"]
                yield SimpleNamespace(
                    prompt_token_ids=[7, 8],
                    outputs=[SimpleNamespace(token_ids=list(range(17)), text="")],
                    metrics=SimpleNamespace(
                        arrival_time=10.0,
                        first_token_time=10.1,
                        finished_time=11.7,
                        last_token_time=11.7,
                    ),
                )

        runtime = FakeVllmEngine()
        engine = ScriptInferenceEngine.__new__(ScriptInferenceEngine)
        engine.model_cfg = {
            "backend": "vllm",
            "generation_contract": "fixed_length_greedy_v1",
        }
        engine.engine = runtime
        engine.backend = "vllm"
        engine.device_id = 0
        engine._counter = 0
        engine._lock = asyncio.Lock()
        engine._reinit_lock = asyncio.Lock()
        engine._engine_dead = False
        engine._lora_in_engine = False
        engine._reinit_attempted = False

        with patch(
            "scripts.run_all_experiments.SamplingParams",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ):
            _, _, output_tokens, timing = asyncio.run(
                engine.generate(
                    prompt="ignored because the plan is canonical",
                    lora_path=None,
                    adapter_id=None,
                    max_tokens=999,
                    input_tokens=999,
                    temperature=0.9,
                    top_p=0.2,
                    _prepared_request=RequestExecutionPlan(
                        prompt="canonical prompt",
                        input_tokens=2,
                        max_tokens=17,
                    ),
                    return_timing=True,
                )
            )

        sampling = runtime.sampling_params
        self.assertEqual(sampling.temperature, 0.0)
        self.assertEqual(sampling.top_p, 1.0)
        self.assertEqual(sampling.max_tokens, 17)
        self.assertTrue(sampling.ignore_eos)
        self.assertEqual(sampling.stop, [])
        self.assertEqual(output_tokens, 17)
        self.assertEqual(
            timing["completion_token_ids_sha256"],
            hashlib.sha256(
                json.dumps(list(range(17)), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        )

    def test_fixed_contract_request_preparation_fails_closed(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._generation_contract = "fixed_length_greedy_v1"
        runner.wl_cfg = {
            "generation_contract": "fixed_length_greedy_v1",
            "fixed_output_max_tokens": 256,
        }
        engine = SimpleNamespace(
            model_cfg={"max_output_tokens_cap": 256, "max_input_len": 759},
            prepare_request=Mock(side_effect=ValueError("tokenizer unavailable")),
        )
        trace = SimpleNamespace(
            prompt="canonical",
            chat_messages=None,
            prompt_input_tokens=4,
            expected_input_tokens=4,
            expected_output_tokens=8,
            prompt_output_tokens=8,
        )

        with self.assertRaisesRegex(RuntimeError, "fallback.*forbidden"):
            runner._prepare_request_execution_plan(engine, trace, 128)

    def test_v2_local_sim_reserves_exact_bytes_before_hardlink(self) -> None:
        class RecordingLimiter:
            def __init__(self, destination: Path) -> None:
                self.destination = destination
                self.sizes = []

            async def throttle(self, size_bytes: int) -> float:
                self.assert_destination_absent()
                self.sizes.append(size_bytes)
                return 7.5

            def assert_destination_absent(self) -> None:
                if self.destination.exists():
                    raise AssertionError("materialized before aggregate reservation")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "remote" / "adapter-a"
            destination = root / "nvme" / "adapter-a"
            source.mkdir(parents=True)
            (source / "adapter_model.bin").write_bytes(b"a" * 101)
            (source / "adapter_config.json").write_bytes(b"b" * 23)

            runner = ScenarioRunner.__new__(ScenarioRunner)
            runner._remote_artifact_client = None
            runner.remote_dir = root / "remote"
            runner.adapter_info = {"adapter-a": {"size_mb": 999.0}}
            runner._local_sim_materialization_mode = "reserve_then_hardlink_v1"
            runner._local_sim_materialization_counts = defaultdict(int)
            limiter = RecordingLimiter(destination)
            runner._bandwidth_limiter = limiter

            ok, elapsed_ms = asyncio.run(
                runner._materialize_remote_adapter_async("adapter-a", destination)
            )

            self.assertTrue(ok)
            self.assertEqual(limiter.sizes, [_path_size_bytes(source)])
            self.assertEqual(limiter.sizes, [124])
            self.assertTrue(destination.exists())
            self.assertGreaterEqual(elapsed_ms, 7.5)
            self.assertEqual(runner._local_sim_materialization_counts["hardlink"], 1)
            self.assertEqual(
                os.stat(source / "adapter_model.bin").st_ino,
                os.stat(destination / "adapter_model.bin").st_ino,
            )

    @staticmethod
    def _contract_request(**overrides) -> RequestResult:
        values = {
            "request_id": "req-1",
            "adapter_id": "adapter-a",
            "is_burst": False,
            "burst_phase": "normal",
            "cache_hit": True,
            "cache_tier": "gpu",
            "lora_io_ms": 0.0,
            "vllm_ttft_ms": 10.0,
            "ttft_ms": 10.0,
            "contention_ms": 0.0,
            "defer_ms": 0.0,
            "tpot_ms": 2.0,
            "e2e_ms": 40.0,
            "input_tokens": 2,
            "output_tokens": 17,
            "cost_usd": 0.0,
            "success": True,
            "generation_contract": "fixed_length_greedy_v1",
            "source_expected_output_tokens": 33,
            "requested_completion_tokens": 17,
            "completion_tokens": 17,
            "completion_token_source": "vllm_token_ids",
            "completion_token_ids_sha256": "tokens-prime",
            "canonical_prompt_sha256": "prompt-map-sha",
            "canonical_prompt_tokens": 2,
            "scheduled_arrival_offset_s": 0.25,
            "output_contract_match": True,
        }
        values.update(overrides)
        return RequestResult(**values)

    def test_request_result_contract_fields_and_request_map_sha_are_auditable(self) -> None:
        prime_request = self._contract_request()
        slora_request = self._contract_request(
            completion_token_source="slora_native_sse_token_ids",
            completion_token_ids_sha256="tokens-slora",
        )
        prime = ScenarioResult("prime", "faaslora_full", 1, requests=[prime_request])
        slora = ScenarioResult("slora", "slora_style", 1, requests=[slora_request])

        expected_rows = [
            {
                "request_id": "req-1",
                "adapter_id": "adapter-a",
                "arrival_time_s": 0.25,
                "source_expected_output_tokens": 33,
                "requested_completion_tokens": 17,
                "canonical_prompt_sha256": "prompt-map-sha",
                "canonical_prompt_tokens": 2,
            }
        ]
        expected_sha = hashlib.sha256(
            json.dumps(
                expected_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        self.assertEqual(prime_request.generation_contract, "fixed_length_greedy_v1")
        self.assertEqual(prime_request.completion_tokens, 17)
        self.assertTrue(prime_request.output_contract_match)
        self.assertEqual(_generation_contract_request_map_sha256(prime), expected_sha)
        self.assertEqual(
            _generation_contract_request_map_sha256(prime),
            _generation_contract_request_map_sha256(slora),
        )

        changed_target = ScenarioResult(
            "changed",
            "faaslora_full",
            1,
            requests=[self._contract_request(requested_completion_tokens=16)],
        )
        self.assertNotEqual(
            _generation_contract_request_map_sha256(prime),
            _generation_contract_request_map_sha256(changed_target),
        )


class RevisionV2ExperimentStackTests(unittest.TestCase):
    def test_async_ensure_local_tuple_preserves_remote_tier_and_io_time(self) -> None:
        with TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter-a"
            adapter_path.mkdir()

            stack = ExperimentStack.__new__(ExperimentStack)
            stack.sync_local_tier_paths = lambda **_kwargs: None
            stack._host_paths = {}
            stack._nvme_paths = {}
            stack._repair_adapter_dir = Mock()
            stack.registry = SimpleNamespace(update_artifact=Mock())
            stack.residency_manager = SimpleNamespace(
                admit_artifact=AsyncMock(return_value=True),
                get_tier_status=lambda _tier: {"artifacts": {"details": []}},
            )

            coordinator = SimpleNamespace(
                _residency_manager=None,
                compute_faaslora_nvme_load_ms=lambda _size_mb: 7.0,
                request_lora_load=AsyncMock(return_value=(2.0, 3.0)),
            )
            ensure_local = AsyncMock(return_value=(str(adapter_path), 125.0))

            resolved = asyncio.run(
                stack.resolve_lora(
                    "adapter-a",
                    30.0,
                    False,
                    ensure_local,
                    coordinator=coordinator,
                )
            )

            self.assertEqual(
                resolved,
                (str(adapter_path), "remote", 132.0, 2.0, 3.0),
            )
            ensure_local.assert_awaited_once_with("adapter-a")
            stack.registry.update_artifact.assert_called_once_with(
                "adapter-a",
                {"storage_path": str(adapter_path)},
            )
            stack.residency_manager.admit_artifact.assert_awaited_once_with(
                "adapter-a",
                StorageTier.NVME,
            )


if __name__ == "__main__":
    unittest.main()
