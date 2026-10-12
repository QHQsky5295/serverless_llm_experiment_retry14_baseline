"""Original factory integration contracts; fixtures are not native qualification."""
import asyncio
import copy
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.cpu_frontend import CONTRACT, configuration_sha256
from scripts import run_all_experiments as runner
from scripts import ieee_tc_preflight as guard

UUID = 'GPU-01234567-0123-4567-89ab-0123456789ab'


def model():
    return dict(backend='vllm', tensor_parallel_size=1, device_id=0,
        visible_device_ids=[0, 1], enforce_eager=True, ieee_gpu_references=True,
        ieee_physical_allocation=True,
        timing_contract='ieee_tc_native_v1', generation_contract='fixed_length_greedy_v1')


def factory():
    return runner.DedicatedScenarioEngineFactory(model_cfg=model(), host_visible_ids=[0, 1],
        cost_cfg={}, coord_cfg={'ieee_cpu_frontend_policy':CONTRACT}, hardware_cfg={},
        coordination_enabled=True, use_subprocess=True)


class FactoryLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_failure_still_joins_cpu_cleanup_and_preserves_failure(self):
        f = factory()
        f.pool = NS(closing=False)
        f.close_preparation = AsyncMock()
        service = object.__new__(runner.ScenarioRunner)
        service.engine_factory = f
        async def runtime_failure():
            self.assertTrue(f.pool.closing)
            raise ValueError('owned runtime failure')
        service._shutdown_instance_pool_runtimes = runtime_failure
        with self.assertRaisesRegex(ValueError, 'owned runtime failure'):
            await service._shutdown_instance_pool()
        f.close_preparation.assert_awaited_once()

    async def test_cancelled_shutdown_waits_for_cpu_owner(self):
        f = factory()
        entered, release = asyncio.Event(), asyncio.Event()
        async def close():
            entered.set()
            await release.wait()
        f.close_preparation = close
        service = object.__new__(runner.ScenarioRunner)
        service.engine_factory = f
        service._shutdown_instance_pool_runtimes = AsyncMock()
        task = asyncio.create_task(service._shutdown_instance_pool())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_notice_prepares_existing_devices_and_shared_factory_claims(self):
        f = factory()
        tickets = [NS(abandon=AsyncMock()), NS(abandon=AsyncMock())]
        with patch.object(runner.SubprocessInferenceEngineProxy, 'prepare_cpu_frontend',
                          side_effect=tickets) as prepare:
            notice = time.monotonic()
            f.start_preparation(deployment_notice_s=notice, devices=[0, 1])
            self.assertEqual(prepare.call_count, 2)
            for i, call in enumerate(prepare.call_args_list):
                self.assertEqual(call.kwargs['runtime_gpu_ids'], [i])
                self.assertEqual(call.kwargs['model_cfg']['device_id'], i)
                self.assertEqual(call.kwargs['deployment_notice_s'], notice)
            with patch.object(runner, '_spawn_dedicated_scenario_engine', AsyncMock(return_value='engine')) as spawn:
                self.assertEqual(await f(device_id=1), 'engine')
                self.assertIs(spawn.await_args.kwargs['cpu_frontend_pool'], f.pool)
            f.stop_preparation()
            self.assertTrue(f.pool.closing)
            await f.close_preparation()
            for t in tickets:
                t.abandon.assert_awaited_once()
            await f.close_preparation()
            self.assertEqual(sum(t.abandon.await_count for t in tickets), 2)

    async def test_failure_is_retained_no_fresh_worker_fallback(self):
        f = factory()
        f.pool = NS(failure=None)
        with patch.object(runner, '_spawn_dedicated_scenario_engine', AsyncMock(side_effect=RuntimeError('startup'))) as spawn:
            with self.assertRaisesRegex(RuntimeError, 'startup'):
                await f(device_id=0)
            self.assertIsInstance(f.pool.failure, RuntimeError)
            self.assertEqual(spawn.await_count, 1)

    async def test_partial_producer_failure_is_owned_and_joined(self):
        f = factory()
        record = dict(ticket=None, process=Mock(pid=30), channel=Mock(), birth={'pid': 30})
        f.preparations.append(record)
        with patch('faaslora.cpu_frontend.abandon_cpu_process', AsyncMock()) as abandon:
            await f.close_preparation()
            abandon.assert_awaited_once_with(record['process'], record['channel'], record['birth'],
                                            guard.gpu_process_identity)

    async def test_final_close_does_not_take_activated_physical_ownership(self):
        f = factory()
        active = NS(state='activation_sent', abandon=AsyncMock())
        f.preparations.append(dict(ticket=active, process=Mock(), channel=Mock(), birth={}))
        await f.close_preparation()
        active.abandon.assert_not_awaited()

    async def test_policy_requires_exact_shared_pool_not_ordinary_spawn(self):
        with self.assertRaisesRegex(ValueError, 'owned preparation pool'):
            await factory()(device_id=0)
        with patch.object(runner.SubprocessInferenceEngineProxy, '_startup_cleanup_done', True):
            with self.assertRaisesRegex(ValueError, 'lifecycle controller'):
                await runner.SubprocessInferenceEngineProxy.spawn(
                    model_cfg=dict(model(),ieee_cpu_frontend_policy=CONTRACT),
                    cost_model={}, device_id=0, runtime_gpu_ids=[0])
        f = factory()
        f.arguments['use_subprocess'] = False
        with self.assertRaises(ValueError):
            f.start_preparation(deployment_notice_s=time.monotonic(), devices=[0])


class Producer(unittest.TestCase):
    def test_actual_candidate_preserves_strict_measured_native_profiles(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        current=yaml.safe_load((root/'configs/ieee_tc/20261012_d471_7b_cpu_frontend_validation_100.yaml').read_text())
        parent=yaml.safe_load((root/'configs/ieee_tc/20261012_d464_7b_host_queue_validation_100.yaml').read_text())
        self.assertEqual(current['model'],parent['model'])
        controller=dict(current['resource_coordination'])
        self.assertEqual(controller.pop('ieee_cpu_frontend_policy'),CONTRACT)
        self.assertEqual(controller,parent['resource_coordination'])
        service=object.__new__(runner.ScenarioRunner)
        service.model_cfg=current['model']
        service.coord_cfg=current['resource_coordination']
        service._initial_runtime_pending=True
        # Exercise the actual runtime/profile binding that rejected D470, not
        # an approximate model dict or a mock that bypasses the loader.
        service._service_profiles=service._load_ieee_service_profiles()
        self.assertIsNotNone(service._service_profiles)
        self.assertIsNotNone(service._load_ieee_preparation_profiles())
        before=runner._prepare_dedicated_subprocess_model_cfg(parent['model'],device_id=0,runtime_gpu_ids=[0])[0]
        self.assertEqual(service._planned_ieee_runtime_model_config(),before)

    def test_fresh_fixed_uuid_and_fd_are_set_before_popen_no_gpu_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identities = lambda pid: dict(pid=pid, start_ticks=pid, uid=os.getuid(),
                cgroup='/service', affinity=[4])
            retained = []
            proc = Mock(pid=123456, poll=Mock(return_value=None))
            census = Mock(sample=Mock(return_value=dict(devices=[dict(index=1, gpu_uuid=UUID)])))
            with patch.dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    FAASLORA_TC_NVML_BINDING='/binding', FAASLORA_TC_NVML_SHA256='hash'), \
                 patch.object(guard, 'verify_current_service', return_value=dict(service_identity={'path': '/service'})), \
                 patch.object(guard, 'load_nvml_binding'), \
                 patch.object(guard, 'NativeGPUCensus', return_value=census), \
                 patch.object(guard, 'gpu_process_identity', side_effect=identities), \
                 patch.object(runner.tempfile, 'mkdtemp', return_value=str(root)), \
                 patch.object(runner.subprocess, 'Popen', return_value=proc) as popen, \
                 patch('faaslora.metrics.metrics_collector.PhysicalGPUAllocation') as allocate:
                ticket = runner.SubprocessInferenceEngineProxy.prepare_cpu_frontend(
                    model_cfg=model(), cost_model={}, device_id=1, runtime_gpu_ids=[1],
                    deployment_notice_s=time.monotonic(), retained=retained, policy=CONTRACT)
                args = popen.call_args
                self.assertEqual(args.kwargs['env']['CUDA_VISIBLE_DEVICES'], UUID)
                self.assertEqual(args.kwargs['env']['FAASLORA_VISIBLE_DEVICES'], UUID)
                self.assertTrue(args.kwargs['close_fds'])
                self.assertTrue(args.kwargs['start_new_session'])
                self.assertEqual(len(args.kwargs['pass_fds']), 1)
                self.assertIn('--activation-fd', args.args[0])
                self.assertIs(retained[0]['ticket'], ticket)
                self.assertIs(retained[0]['process'], proc)
                self.assertFalse((root/'ready.json').exists())
                allocate.assert_not_called()
                census.close.assert_called_once()
                ticket.channel.close()

    def test_postspawn_birth_failure_retains_process_and_closes_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            retained = []
            proc = Mock(pid=123456)
            own = dict(pid=os.getpid(), start_ticks=1, cgroup='/service')
            with patch.dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn',
                    FAASLORA_TC_NVML_BINDING='/binding', FAASLORA_TC_NVML_SHA256='hash'), \
                 patch.object(guard, 'verify_current_service', return_value=dict(service_identity={'path': '/service'})), \
                 patch.object(guard, 'load_nvml_binding'), \
                 patch.object(guard, 'NativeGPUCensus', return_value=Mock(
                     sample=Mock(return_value={'devices': [dict(index=0, gpu_uuid=UUID)]}))), \
                 patch.object(guard, 'gpu_process_identity', side_effect=[own, None]), \
                 patch.object(runner.tempfile, 'mkdtemp', return_value=tmp), \
                 patch.object(runner.subprocess, 'Popen', return_value=proc):
                with self.assertRaises(ValueError):
                    runner.SubprocessInferenceEngineProxy.prepare_cpu_frontend(
                        model_cfg=model(), cost_model={}, device_id=0, runtime_gpu_ids=[0],
                        deployment_notice_s=time.monotonic(), retained=retained, policy=CONTRACT)
            self.assertIs(retained[0]['process'], proc)
            self.assertIsNone(retained[0]['ticket'])
            self.assertEqual(retained[0]['channel'].fileno(), -1)


class NativeFactoryBinding(unittest.IsolatedAsyncioTestCase):
    async def test_release_and_helper_exit_precede_replacement(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = Mock(pid=20, returncode=0, poll=Mock(return_value=0))
            allocation = NS(wait_workers=AsyncMock(side_effect=lambda **k: events.append('native_exit')),
                            release=Mock(side_effect=lambda: events.append('physical_return')))
            proxy = runner.SubprocessInferenceEngineProxy(process=process, host='localhost', port=1,
                model_cfg=model(), cost_model={}, device_id=0, workdir=root,
                log_path=root/'worker.log', physical_allocation=allocation)
            ticket = NS(evidence={'birth': {'pid':20}}, identity=Mock())
            proxy._cpu_frontend_ticket = ticket
            proxy._cpu_frontend_pool = Mock(returned=Mock(side_effect=lambda *a, **k: events.append('replace')))
            proxy._terminate_process_tree = AsyncMock()
            proxy._close_all_rpc_channels = AsyncMock()
            with patch('faaslora.cpu_frontend.finish_process_group',
                       AsyncMock(side_effect=lambda *a: events.append('helpers_exit'))):
                await proxy.shutdown()
            self.assertEqual(events, ['native_exit','physical_return','helpers_exit','replace'])
            self.assertIsNone(proxy._cpu_frontend_ticket)

    async def test_actual_spawn_binds_before_permission_and_never_launches_second_process(self):
        from faaslora.metrics import metrics_collector
        with tempfile.TemporaryDirectory() as tmp:
            root, events = Path(tmp), []
            local, updates = runner._prepare_dedicated_subprocess_model_cfg(model(), device_id=0, runtime_gpu_ids=[0])
            birth = dict(pid=123456, start_ticks=2, cgroup='/service')
            prep = dict(configuration_sha256=configuration_sha256(local, {}), birth=birth)
            process = Mock(pid=123456, poll=Mock(return_value=None))
            ticket = NS(device_id=0, evidence=prep, process=process,
                launch_inputs=dict(root=root, model_cfg=local, updates=updates, env={}, payload={}),
                activate=Mock(side_effect=lambda allocation: events.append('activate')))
            pool = Mock(claim=Mock(return_value=ticket))
            allocation = Mock(gpu_uuids=[UUID], owner_id='lease',
                bind_process=Mock(side_effect=lambda *a, **k: events.append('bind')))
            def allocate(**kwargs):
                events.append('acquire')
                self.assertEqual(kwargs['device_indices'], [0])
                return allocation
            (root/'ready.json').write_text(json.dumps(dict(status='ready',host='127.0.0.1',port=10,configuration_contract='fixture')))
            (root/'cpu_prepared.json').write_text(json.dumps(dict(preparation=prep, engine_ready=False)))
            (root/'cpu_activated.json').write_text(json.dumps(dict(permission=dict(lease_id='lease', birth=birth))))
            read = Path.read_text
            def observed_read(path, *args, **kwargs):
                events.append(path.name)
                return read(path, *args, **kwargs)
            with patch.object(runner.SubprocessInferenceEngineProxy, '_startup_cleanup_done', True), \
                 patch.object(runner.SubprocessInferenceEngineProxy, 'ieee_worker_observation', AsyncMock(return_value={'workers': ['native']})), \
                 patch.object(guard, 'verify_current_service', return_value=dict(service_identity={'path':'/service'})), \
                 patch.object(guard, 'load_nvml_binding'), patch.object(guard, 'NativeGPUCensus'), \
                 patch.object(guard, 'gpu_process_identity', return_value=birth), \
                 patch.object(metrics_collector, 'PhysicalGPUAllocation', side_effect=allocate), \
                 patch('faaslora.runtime_configuration.initialized_worker_configuration', return_value=local), \
                 patch.dict(os.environ, FAASLORA_TC_LAUNCH_RECEIPT=str(root/'launch.json'),
                     FAASLORA_TC_NVML_BINDING='/binding', FAASLORA_TC_NVML_SHA256='hash'), \
                 patch.object(Path, 'read_text', observed_read), \
                 patch.object(runner.subprocess, 'Popen') as popen:
                result = await runner.SubprocessInferenceEngineProxy.spawn(
                    model_cfg=model(), cost_model={}, device_id=0, runtime_gpu_ids=[0],
                    cpu_frontend_pool=pool)
            self.assertEqual(events[:3], ['acquire','bind','activate'])
            self.assertGreater(events.index('cpu_prepared.json'), events.index('acquire'))
            self.assertIs(result._process, process)
            self.assertIs(result._cpu_frontend_ticket, ticket)
            self.assertIs(ticket.physical_allocation, allocation)
            self.assertEqual(allocation.bind_process.call_args.kwargs['preparation'], prep)
            allocation.confirm_workers.assert_called_once_with(['native'])
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
