"""Frozen open-loop transport tests; fixtures are not experimental workloads."""
import asyncio
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

from faaslora.clock import local_monotonic_clock_id
from faaslora.datasets.workload_generator import (
    FrozenReplayPlan, ExternalReplayIngress, publish_frozen_replay,
    render_role_lines, canonical_fixed_prompt, prepare_frozen_http_request, replay_frozen_http,
)


class FrozenViews(unittest.TestCase):
    def load_rows(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.json'
            path.write_text(json.dumps({'requests': rows}))
            return FrozenReplayPlan.load(path, **kwargs)

    def test_sort_identity_and_content_hash_without_rewriting_source(self):
        rows = [{'request_id':'b', 'arrival_time_s':12, 'body':{'prompt':'second'}},
                {'request_id':'a', 'arrival_time_s':10, 'body':{'prompt':'first'}}]
        plan = self.load_rows(rows)
        self.assertEqual([e.request_id for e in plan.entries], ['a', 'b'])
        self.assertEqual([e.offset_s for e in plan.entries], [0, 2])
        self.assertEqual(json.loads(plan.entries[0].source_json), rows[1])
        self.assertEqual(plan.identity()['source_count'], 2)

    def test_w1_exact_phase_formula_not_service_drain(self):
        rows = [{'request_id':str(i), 'arrival_time_s':2*i+10} for i in range(1002)]
        w1 = self.load_rows(rows, profile='W1')
        self.assertEqual([w1.entries[i].offset_s for i in (0, 499, 500, 999, 1000, 1001)],
                         [0, 499, 529, 1028, 1058, 1059])

    def test_prefix_and_rate_are_declared_views_not_new_input(self):
        rows = [{'request_id':str(i), 'arrival_time_s':2*i} for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.json'
            path.write_text(json.dumps({'requests': rows}))
            a = FrozenReplayPlan.load(path)
            b = FrozenReplayPlan.load(path, count=2, rate_scale=8.)
            self.assertEqual(a.source_sha256, b.source_sha256)
            self.assertNotEqual(a.identity()['view_sha256'], b.identity()['view_sha256'])
            self.assertEqual(b.entries[-1].offset_s, .25)
            self.assertEqual(b.identity()['source_count'], 4)

    def test_invalid_trace_or_transform_rejected(self):
        row = {'request_id':'one', 'arrival_time_s':0}
        for rows, kwargs in [([], {}), ([row,row], {}), ([dict(row, arrival_time_s=math.nan)], {}),
                             ([row], {'rate_scale':0}), ([row], {'count':2}),
                             ([row], {'profile':'W2'})]:
            with self.assertRaises(ValueError):
                self.load_rows(rows, **kwargs)


class CanonicalHTTPInput(unittest.TestCase):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return ([1] if add_special_tokens else []) + [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(c) for c in ids)

    def test_role_rendering_and_special_tokens_have_distinct_identity(self):
        prompt = render_role_lines([dict(role='user', content='hello')])
        self.assertEqual(prompt, 'User: hello')
        row = canonical_fixed_prompt(prompt, self.Tokenizer(), 256)
        self.assertEqual(row['canonical_prompt_tokens'], 11)
        self.assertEqual(row['input_token_ids'], [1]+list(map(ord, prompt)))
        self.assertNotEqual(row['canonical_prompt_sha256'], row['native_prompt_token_ids_sha256'])

    def test_common_tail_cap_and_bad_targets(self):
        row = canonical_fixed_prompt('a'*1000, self.Tokenizer(), 256)
        self.assertEqual(len(row['input_token_ids']), 760)
        for target in (0, True, 1.5, 257):
            with self.assertRaises(ValueError):
                canonical_fixed_prompt('x', self.Tokenizer(), target)
        with self.assertRaises(ValueError):
            canonical_fixed_prompt('x', self.Tokenizer(), 256, max_model_len=10)
        with self.assertRaises(ValueError):
            render_role_lines([dict(role='user', content=['not text'])])

    def test_source_adapter_and_target_not_substituted(self):
        from faaslora.datasets.workload_generator import FrozenReplayEntry
        source = dict(expected_output_tokens=999, adapter_id='adapter-a',
                      body=dict(messages=[dict(role='user', content='x')]))
        entry = FrozenReplayEntry('r', 0., json.dumps(source), 'source-sha')
        row = prepare_frozen_http_request(entry, self.Tokenizer(), 'model')
        self.assertEqual(row['body']['max_tokens'], 256)
        self.assertEqual(row['body']['lora_adapter_name'], 'adapter-a')
        self.assertEqual(row['body']['input_tokens'], row['input_token_ids'])
        self.assertEqual(row['body']['stop'], [])
        self.assertTrue(row['body']['ignore_eos'])


class HTTPOpenLoop(unittest.IsolatedAsyncioTestCase):
    async def test_all_arrivals_exist_while_first_response_is_delayed(self):
        from faaslora.datasets.workload_generator import FrozenReplayEntry
        entries = tuple(FrozenReplayEntry(str(i), i*.005, '{}', str(i)) for i in range(5))
        plan = FrozenReplayPlan('fixture', 'sha', entries, 'W0', 1., 5)
        now = time.perf_counter()
        origin = dict(deployment_notice_s=now, replay_t0_s=now+.005,
                      clock_id=local_monotonic_clock_id())
        events, begun, ended = [], [], []
        release = asyncio.Event()

        async def send(row, event):
            begun.append(event['request_id'])
            await release.wait()
            ended.append(event['request_id'])
            return {'ok': True}

        task = asyncio.create_task(replay_frozen_http(plan, origin,
            {e.request_id: {} for e in entries}, send, events.append))
        await asyncio.sleep(.08)
        self.assertEqual(len(begun), 5)
        self.assertEqual(ended, [])
        self.assertEqual([e['planned_arrival_s'] for e in events],
                         [origin['replay_t0_s']+e.offset_s for e in entries])
        release.set()
        self.assertEqual((await task)['N_response'], 5)

    async def test_failure_is_terminal_and_does_not_suppress_offered_requests(self):
        from faaslora.datasets.workload_generator import FrozenReplayEntry
        entries = tuple(FrozenReplayEntry(str(i), 0., '{}', str(i)) for i in range(3))
        plan = FrozenReplayPlan('fixture', 'sha', entries, 'W0', 1., 3)
        now = time.perf_counter()
        origin = dict(deployment_notice_s=now, replay_t0_s=now,
                      clock_id=local_monotonic_clock_id())
        events = []

        async def send(row, event):
            if event['request_id'] == '1':
                raise ValueError('HTTP error')
            return {'ok': True}

        counts = await replay_frozen_http(plan, origin, {e.request_id: {} for e in entries},
                                          send, events.append)
        self.assertEqual(counts, dict(N_plan=3, N_arrived=3, N_terminal=3, N_response=2, N_failed=1))
        self.assertEqual(len([e for e in events if e['event']=='http_request_failed']), 1)

    async def test_expired_arrival_not_given_new_timeout_and_clock_mismatch_refused(self):
        from faaslora.datasets.workload_generator import FrozenReplayEntry
        plan = FrozenReplayPlan('fixture', 'sha', (FrozenReplayEntry('r', 0., '{}', 'sha'),), 'W0', 1., 1)
        now = time.perf_counter()-10
        origin = dict(deployment_notice_s=now, replay_t0_s=now, clock_id=local_monotonic_clock_id())
        async def send(*args):
            self.fail('expired request must not reach endpoint')
        counts = await replay_frozen_http(plan, origin, {'r': {}}, send, lambda e: None, request_timeout_s=1)
        self.assertEqual(counts['N_failed'], 1)
        with self.assertRaisesRegex(ValueError, 'clock/input'):
            await replay_frozen_http(plan, dict(origin, clock_id='other-host'), {'r': {}}, send, lambda e: None)


class OpenLoopTransport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ptcr-')
        self.root = Path(self.tmp.name)
        path = self.root/'trace.json'
        path.write_text(json.dumps({'requests':[
            {'request_id':str(i), 'arrival_time_s':i*.01, 'adapter_id':'adapter-a',
             'body':{'prompt':'fixture'}} for i in range(5)]}))
        self.plan = FrozenReplayPlan.load(path)
        self.events = []
        now = time.perf_counter()
        self.origin = {'deployment_notice_s':now, 'replay_t0_s':now+.005,
                       'clock_id':local_monotonic_clock_id()}
        self.context = {**self.origin, 'plan':self.plan.identity(), 'nonce':'test',
                        'address':str(self.root/'socket'), 'frame_limit':16384}

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def start_publisher(self):
        async def origin():
            return self.origin
        task = asyncio.create_task(publish_frozen_replay(self.plan, self.context['address'],
                                                       'test', origin, self.events.append))
        while not self.events:
            await asyncio.sleep(.001)
        return task

    async def test_late_service_does_not_shift_arrivals_or_clock_origin(self):
        publisher = await self.start_publisher()
        await asyncio.sleep(.08)
        created = [e for e in self.events if e['event']=='request_created']
        self.assertEqual(len(created), 5)
        self.assertFalse(any(e['event']=='request_submitted' for e in self.events))
        ingress = ExternalReplayIngress(self.plan, self.context)
        async for _, _ in ingress.receive():
            await asyncio.sleep(.005)
        counts = await publisher
        self.assertEqual(counts, {'N_plan':5, 'N_arrived':5, 'N_submitted':5})
        self.assertTrue(ingress.complete)
        for i, e in enumerate(created):
            self.assertEqual(e['planned_arrival_s'], self.origin['replay_t0_s']+i*.01)
            r = ingress.records[str(i)]
            self.assertLessEqual(r['task_created_s'], r['client_submit_s'])
            self.assertLessEqual(r['client_submit_s'], r['server_received_s'])
        self.assertGreater(ingress.records['0']['server_received_s']-self.origin['replay_t0_s'], .06)

    async def test_reception_during_startup_precedes_service_consumption(self):
        publisher = await self.start_publisher()
        events = []
        ingress = ExternalReplayIngress(self.plan, self.context, emit=events.append)
        await ingress.start()
        await asyncio.sleep(.08)  # Startup pending, no request consumer yet.
        self.assertEqual(len(ingress.records), 5)
        self.assertTrue(ingress.complete)
        self.assertFalse(any(e['event']=='request_dequeued' for e in events))
        ready = time.perf_counter()
        async for i, record in ingress.receive():
            self.assertLess(record['server_received_s'], ready)
            self.assertGreaterEqual(record['service_dequeued_s'], ready)
        await publisher
        await ingress.close()
        self.assertEqual(sum(e['event']=='request_received' for e in events), 5)
        self.assertEqual(sum(e['event']=='request_dequeued' for e in events), 5)
        terminal = [e for e in events if e['event']=='service_ingress_terminal']
        self.assertEqual(len(terminal), 1)
        self.assertTrue(terminal[0]['complete'])

    async def test_observer_gets_prior_received_history_once_then_live_arrivals(self):
        publisher = await self.start_publisher()
        ingress = ExternalReplayIngress(self.plan, self.context)
        await ingress.start()
        while len(ingress.records) < 2:
            await asyncio.sleep(.001)
        observed = []
        ingress.subscribe(observed.append)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            ingress.subscribe(observed.append)
        async for _ in ingress.receive():
            pass
        self.assertEqual([r['request_id'] for r in observed], ['0','1','2','3','4'])
        self.assertNotIn('service_dequeued_s', observed[0])
        self.assertEqual([r['server_received_s'] for r in observed], ingress.observed_times)
        with self.assertRaisesRegex(RuntimeError, 'one service consumer'):
            async for _ in ingress.receive():
                pass
        await publisher
        await ingress.close()

    async def test_close_during_startup_preserves_observed_prefix(self):
        publisher = await self.start_publisher()
        events = []
        ingress = ExternalReplayIngress(self.plan, self.context, emit=events.append)
        prefix_received = asyncio.Event()
        release_tail = asyncio.Event()
        read_transport = ingress._read_transport

        async def held_transport():
            # Exercise the real socket and packet validation, but stop reception
            # at a known prefix. start() is not a guarantee that a short replay
            # has not already finished by the time its caller resumes.
            source = read_transport()
            try:
                async for item in source:
                    yield item
                    prefix_received.set()
                    await release_tail.wait()
            finally:
                await source.aclose()

        ingress._read_transport = held_transport
        try:
            await ingress.start()
            await prefix_received.wait()
            self.assertEqual(list(ingress.records), ['0'])
            await ingress.close()
            self.assertEqual(events[-1]['event'], 'service_ingress_terminal')
            self.assertEqual(events[-1]['N_plan'], 5)
            self.assertFalse(events[-1]['complete'])
            self.assertEqual(events[-1]['N_received'], 1)
            self.assertEqual(events[-1]['N_received'], len(ingress.records))
        finally:
            await ingress.close()
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)

    async def test_clock_or_plan_mismatch_is_not_silent_fallback(self):
        for changed in (dict(self.context, clock_id='another-host'),
                        dict(self.context, plan=dict(self.plan.identity(), count=4))):
            with self.assertRaisesRegex(ValueError, 'trace/view/clock'):
                ExternalReplayIngress(self.plan, changed)

    async def test_cancel_keeps_full_planned_denominator(self):
        publisher = await self.start_publisher()
        publisher.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await publisher
        final = self.events[-1]
        self.assertEqual(final['event'], 'replay_incomplete')
        self.assertEqual(final['N_plan'], 5)
        self.assertEqual(final['N_submitted'], 0)
        self.assertLess(final['N_arrived'], 5)

    async def test_early_or_changed_payload_is_rejected(self):
        async def bad_handler(reader, writer):
            await reader.readline()
            writer.write((json.dumps({'event':'replay_header', **self.origin,
                                     'plan':self.plan.identity()})+'\n').encode())
            writer.write((json.dumps({'event':'request', 'index':0, 'request_id':'0',
                'source_item_sha256':self.plan.entries[0].source_sha256,
                'source_request':{'body':{'prompt':'wrong'}},
                'planned_arrival_s':self.origin['replay_t0_s'],
                'task_created_s':self.origin['replay_t0_s'],
                'client_submit_s':self.origin['replay_t0_s']})+'\n').encode())
            await writer.drain()
            writer.close()
        server = await asyncio.start_unix_server(bad_handler, self.context['address'])
        try:
            ingress = ExternalReplayIngress(self.plan, self.context)
            with self.assertRaisesRegex(ValueError, 'changed content'):
                async for _ in ingress.receive():
                    pass
            self.assertFalse(ingress.complete)
            self.assertEqual(len(ingress.records), 0)
        finally:
            server.close()
            await server.wait_closed()


if __name__ == '__main__':
    unittest.main()
