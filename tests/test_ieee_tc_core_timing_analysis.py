"""Exact host-time partition fixtures; not new performance experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import (summarize_core_host_spans,
    core_interval_overlap, analyze_core_host_timing, summarize_local_file_lock_spans,
    analyze_local_file_lock_timing, join_local_publication_spans)


class LocalFileLockAnalysis(unittest.TestCase):
    @staticmethod
    def fixture():
        def row(seq, parent, operation, a, b, c, cpu, acquired=True, thread=9):
            value = dict(event='file_owner_local_lock', sequence=seq, parent_sequence=parent,
                depth=0 if parent is None else 1, operation=operation,
                caller_file='owner.py', caller_line=1, identifiers={'adapter_id':'a'},
                owner_id='owner', pid=7, thread_ident=thread, native_thread_id=thread,
                clock='host_monotonic_ns', acquired=acquired, error_type=None,
                attempt_started_ns=a, ended_ns=c,
                attempt_thread_cpu_ns=cpu[0], ended_thread_cpu_ns=cpu[2])
            if acquired:
                value.update(acquired_ns=b, acquired_thread_cpu_ns=cpu[1])
            return value
        return [row(2, 1, 'snapshot', 3, 4, 9, (2,3,5)),
            row(3, 1, 'try_read', 10, None, 12, (6,None,7), acquired=False),
            row(1, None, 'publish', 0, 2, 20, (0,1,10)),
            row(4, None, 'snapshot', 18, 21, 25, (0,1,3), thread=11),
            dict(event='file_owner_timing_end_v1', owner_closed=False, records=4,
                submitted_commands=4, observed_receipts=4, truncated=False, observer_ns=2)]

    def test_exact_nonadditive_hold_acquisition_and_exclusive_partition(self):
        rows = self.fixture(); before = copy.deepcopy(rows)
        result = summarize_local_file_lock_spans(rows)
        self.assertEqual(rows, before)
        self.assertTrue(result['complete'])
        self.assertFalse(result['owner_closed'])
        self.assertFalse(result['causal_attribution_established'])
        by_id = {r['sequence']:r for r in result['records']}
        self.assertEqual(by_id[1]['hold_ns'], 18)
        self.assertEqual(by_id[1]['exclusive_hold_ns'], 10)
        self.assertEqual(by_id[1]['exclusive_thread_cpu_ns'], 5)
        self.assertEqual(by_id[3]['acquisition_ns'], 2)
        self.assertEqual(result['index']['hold']['total_ns'], 22)
        self.assertEqual(result['index']['acquisition']['total_ns'], 8)
        self.assertEqual(core_interval_overlap(result['index']['hold'], 19, 23), 3)
        group = next(g for g in result['groups'] if g['scope']=='all_inclusive' and g['operation']=='snapshot')
        self.assertEqual(group['hold_ns']['p95_type1_ms'], 5/1e6)
        root = next(g for g in result['groups'] if g['scope']=='root_only' and g['operation']=='snapshot')
        self.assertEqual(root['count'], 1)

    def test_missing_close_and_truncation_are_not_zero_or_complete(self):
        result = summarize_local_file_lock_spans(self.fixture()[:-1])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['truncated'])
        self.assertIsNone(result['observer_ns'])
        rows = self.fixture()
        rows = [rows[0], dict(rows[-1], records=1, truncated=True)]
        result = summarize_local_file_lock_spans(rows)
        self.assertFalse(result['complete'])
        self.assertEqual(result['orphan_sequence_ids'], [2])
        self.assertFalse(result['records'][0]['exclusive_complete'])

    def test_incomplete_submitted_acquisition_is_retained(self):
        rows = self.fixture(); rows[-1]['submitted_commands'] = 5
        result = summarize_local_file_lock_spans(rows)
        self.assertFalse(result['complete'])
        self.assertEqual(result['observation_count'], 4)

    def test_close_counts_identity_and_parent_errors_are_rejected(self):
        changes = [lambda r:r[-1].update(records=3),
            lambda r:r[-1].update(observed_receipts=5),
            lambda r:r[0].update(sequence=True),
            lambda r:r[0].update(parent_sequence=9),
            lambda r:r[0].update(thread_ident=99),
            lambda r:r[0].update(depth=0),
            lambda r:r[0].update(owner_id='other'),
            lambda r:r[0].update(ended_ns=21),
            lambda r:r[0].update(acquired_ns=2),
            lambda r:r[0].update(acquired_thread_cpu_ns=9),
            lambda r:r[0].update(identifiers={'unknown':'payload'}),
            lambda r:r[0].update(clock='walltime'),
            lambda r:r[3].update(acquired_ns=19)]
        for change in changes:
            with self.subTest(change=change):
                rows = self.fixture(); change(rows)
                with self.assertRaises(ValueError):
                    summarize_local_file_lock_spans(rows)
        with self.assertRaises(ValueError):
            summarize_local_file_lock_spans(self.fixture()+[self.fixture()[0]])

    def test_emission_order_does_not_invent_cross_thread_order(self):
        rows = self.fixture()
        shuffled = [rows[3], rows[0], rows[1], rows[2], rows[-1]]
        self.assertEqual(summarize_local_file_lock_spans(rows),
                         summarize_local_file_lock_spans(shuffled))

    def test_failed_acquire_typed_error_is_counted_without_hold(self):
        rows = self.fixture(); rows[1]['error_type'] = 'ValueError'
        result = summarize_local_file_lock_spans(rows)
        group = next(g for g in result['groups'] if g['operation'] == 'try_read')
        self.assertEqual((group['acquired'], group['raised']), (0,1))
        self.assertEqual(group['hold_ns']['sum_ms'], 0)

    def test_local_file_audit_binds_header_to_launch_and_writes_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journals = root/'diagnostic_file_owner'
            journals.mkdir()
            owner = '/sys/fs/cgroup/test-local-owner.scope'
            header = dict(event='file_owner_timing_journal_v1', pid=7, start_ticks=11,
                local_owner_id='owner', cgroup=owner, clock_id='clock',
                observation_mode='local_lock_spans_v1', formal_performance_result=False,
                observer_overhead_included=True, local_lock_intervals_include_scheduling=True,
                local_thread_cpu_excludes_parallel_hash_readers=True,
                replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=2)
            journal = journals/'7-11-owner.jsonl'
            journal.write_text(''.join(json.dumps(row)+'\n' for row in [header]+self.fixture()))
            receipt = root/'receipt.json'
            receipt.write_text(json.dumps(dict(allow_exec=True, service_identity=dict(path=owner),
                external_replay=dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=2))))
            deployment = root/'deployment.json'
            deployment.write_text(json.dumps(dict(clock_id='clock', plan=dict(count=2))))
            summary, records, indexes = analyze_local_file_lock_timing(journals, receipt, deployment, root/'output')
            self.assertEqual(summary['complete_owners'], 1)
            self.assertEqual(summary['observation_count'], 4)
            self.assertTrue((root/'output/lock_spans.csv').is_file())
            self.assertEqual(len(summary['sources']), 4)
            self.assertEqual(indexes['7-11-owner']['hold']['total_ns'], 22)
            with self.assertRaises(FileExistsError):
                analyze_local_file_lock_timing(journals, receipt, deployment, root/'output')
            header['clock_id'] = 'different-clock'
            journal.write_text(''.join(json.dumps(row)+'\n' for row in [header]+self.fixture()))
            with self.assertRaisesRegex(ValueError, 'identity/clock'):
                analyze_local_file_lock_timing(journals, receipt, deployment, root/'wrong')
            self.assertFalse((root/'wrong').exists())

    def test_publication_joins_exact_transfer_and_owner_not_adapter_or_time(self):
        rows = self.fixture()
        rows[2]['operation'] = 'publish_transfer'
        rows[2]['identifiers']['transfer_id'] = 'transfer'
        spans = summarize_local_file_lock_spans(rows)['records']
        transfer = dict(transfer_id='transfer', artifact_id='a', state='published',
            loading_clock_id='clock', confirmed_file_publication=dict(owner_id='owner'),
            extraction_completed_monotonic_s=0., published_monotonic_s=23e-9)
        joined, = join_local_publication_spans([transfer], spans, clock_id='clock')
        self.assertEqual(joined['status'], 'joined')
        self.assertEqual(joined['hold_ms'], 18/1e6)
        self.assertAlmostEqual(sum(joined[k] for k in ('before_lock_attempt_ms',
            'acquisition_ms','hold_ms','after_lock_release_edge_ms')), joined['publication_envelope_ms'])
        unjoined, = join_local_publication_spans([dict(transfer, transfer_id='other')], spans, clock_id='clock')
        self.assertEqual(unjoined['status'], 'unjoined')
        self.assertNotIn('hold_ms', unjoined)
        with self.assertRaises(ValueError):
            join_local_publication_spans([transfer, transfer], spans, clock_id='clock')
        with self.assertRaises(ValueError):
            join_local_publication_spans([transfer], spans, clock_id='wrong')
        with self.assertRaises(ValueError):
            join_local_publication_spans([dict(transfer, published_monotonic_s=17e-9)], spans, clock_id='clock')
        with self.assertRaises(ValueError):
            join_local_publication_spans([dict(transfer, confirmed_file_publication=dict(owner_id='other'))], spans, clock_id='clock')


def events():
    def row(identity,parent,boundary,a,b,details):
        return dict(event='native_core_host_span_v1',id=identity,parent=parent,boundary=boundary,
            begin_ns=a,end_ns=b,thread_cpu_begin_ns=a//2,thread_cpu_end_ns=b//2,
            outcome='returned',details=details,preceding_observer_ns=0,preceding_write_ns=0)
    return [row(2,1,'client',2,6,dict(request_type='UTILITY',utility='collective_rpc')),
        row(1,None,'input_queue',0,10,dict(unfinished_before=1,running_before=1,waiting_before=0)),
        row(4,3,'schedule',12,15,dict(scheduled_tokens=11,scheduled_requests=2,multi_token_requests=1)),
        row(3,None,'step',11,31,dict(unfinished_before=1,running_before=1,waiting_before=0,model_executed=True)),
        dict(event='observation_close',monotonic_ns=32,completed_spans=4,truncated=False,
             unfinished_span_ids=[],observer_ns=1,write_ns=1,bytes_before_close=0)]


class NativeHostInventoryAnalysis(unittest.TestCase):
    @staticmethod
    def fixture():
        rows=events()
        inventory=copy.deepcopy(rows[0])
        inventory.update(id=3,parent=2,boundary='host_inventory',begin_ns=3,end_ns=4,
            thread_cpu_begin_ns=1,thread_cpu_end_ns=2,details=dict(
                scope='native_registered_tensor_storage_capacity',allocation_count=2,
                adapter_count=1,host_tensor_storage_bytes=64))
        allocator=copy.deepcopy(inventory)
        allocator.update(id=4,boundary='host_allocator_stats',begin_ns=4,end_ns=5,
            thread_cpu_begin_ns=2,thread_cpu_end_ns=2,details={})
        rows[2].update(id=6,parent=5);rows[3]['id']=5
        rows[-1]['completed_spans']=6
        return [inventory,allocator]+rows

    def test_exact_nested_counts_and_original_partition_unchanged(self):
        rows=self.fixture();before=copy.deepcopy(rows)
        with self.assertRaises(ValueError):summarize_core_host_spans(rows)
        value=summarize_core_host_spans(rows,host_inventory=True,include_iterations=True,
                                         include_input_turns=True)
        original=summarize_core_host_spans(events())
        self.assertEqual(rows,before)
        self.assertEqual(value['spans'],6)
        self.assertTrue(value['complete'])
        for key in ('client','input_nonclient','input_queue','step'):
            self.assertEqual(value['index'][key],original['index'][key])
        self.assertEqual(value['index']['host_inventory']['total_ns'],1)
        self.assertEqual(value['index']['host_allocator_stats']['total_ns'],1)
        self.assertEqual(value['host_inventory_records'][0]['allocation_count'],2)
        self.assertEqual(value['host_inventory_records'][0]['root'],1)
        self.assertEqual(value['iterations']['records'][0]['step_ns'],20)
        self.assertEqual(value['input_turns']['records'][0]['client_count'],1)

    def test_direct_root_inventory_does_not_subtract_input_nonclient(self):
        rows=events()
        rows[0].update(boundary='host_inventory',details=self.fixture()[0]['details'])
        value=summarize_core_host_spans(rows,host_inventory=True)
        self.assertEqual(value['index']['input_nonclient']['total_ns'],10)
        self.assertNotIn('client',value['index'])
        self.assertEqual(value['host_inventory_records'][0]['label'],'input_queue')

    def test_scope_fields_times_and_parent_contract_rejected(self):
        changes=[lambda r:r[0]['details'].update(scope='other'),
            lambda r:r[0]['details'].update(adapter_count=True),
            lambda r:r[0]['details'].update(host_tensor_storage_bytes=-1),
            lambda r:r[0]['details'].update(pointer=123),
            lambda r:r[1]['details'].update(count=1),
            lambda r:r[0].update(parent=None),
            lambda r:r[1].update(parent=3),
            lambda r:r[0].update(begin_ns=1),
            lambda r:r[1].update(begin_ns=3),
            lambda r:r[-1].update(completed_spans=4)]
        for change in changes:
            with self.subTest(change=change):
                rows=self.fixture();change(rows)
                with self.assertRaises(ValueError):summarize_core_host_spans(rows,host_inventory=True)

    def test_requested_raised_and_partial_are_not_invented_full_scope(self):
        rows=self.fixture()
        rows[0]['details'].update(scope='native_requested_tensor_storage_capacity',
                                  requested_host_storage_bytes=64)
        rows[0]['details'].pop('host_tensor_storage_bytes')
        value=summarize_core_host_spans(rows,host_inventory=True)
        self.assertEqual(value['host_inventory_records'][0]['storage_bytes'],64)
        self.assertEqual(value['host_inventory_records'][0]['scope'],
                         'native_requested_tensor_storage_capacity')
        rows[0].update(outcome='raised',details={})
        value=summarize_core_host_spans(rows[:-1],host_inventory=True)
        self.assertFalse(value['complete'])
        self.assertEqual(value['host_inventory_records'][0]['scope'],'raised')
        self.assertIsNone(value['host_inventory_records'][0]['storage_bytes'])
        orphan=summarize_core_host_spans(rows[:1],host_inventory=True)
        self.assertEqual(orphan['orphan_span_ids'],[3])
        self.assertIsNone(orphan['host_inventory_records'][0]['root'])

    def test_file_header_and_decode_subsets_do_not_double_count(self):
        import csv
        for flag in ('fresh_inventory_v1','unknown',None):
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as tmp:
                args=NativeCoreAnalysis().write_fixture(Path(tmp))
                header=json.loads(args[5].read_text().splitlines()[0])
                header.update(host_inventory_timing=flag,thread_ident=42,
                    nested_spans_not_additive=True,
                    inventory_scope='owner_thread_calls_inside_observed_core_spans',
                    inventory_timing='whole_function_no_per_tensor_instrumentation')
                args[5].write_text(''.join(json.dumps(r)+'\n' for r in [header]+self.fixture()))
                if flag != 'fresh_inventory_v1':
                    with self.assertRaises(ValueError):analyze_core_host_timing(*args[:5])
                    self.assertFalse(args[4].exists())
                    continue
                result=analyze_core_host_timing(*args[:5])
                self.assertEqual(result['host_inventory_observations'],2)
                self.assertEqual(result['complete_request_observations'],1)
                with (args[4]/'decode_windows.csv').open() as f:row,=csv.DictReader(f)
                self.assertAlmostEqual(float(row['host_allocator_stats_ms']),.000001)
                self.assertEqual(float(row['host_inventory_ms']),0)
                self.assertAlmostEqual(sum(float(row[k]) for k in
                    ('client_ms','input_nonclient_ms','step_ms','other_or_unobserved_ms')),
                    float(row['decode_ms']))
                with (args[4]/'host_inventory.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))),2)


class NativeStepComponentAnalysis(unittest.TestCase):
    def test_explicit_partial_keeps_true_roots_without_inventing_footer(self):
        from scripts.analyze_control_path_overhead import summarize_core_step_components
        rows=self.fixture()
        orphan=copy.deepcopy(rows[2])
        orphan.update(id=6,parent=5,begin_ns=33,end_ns=34,
                      thread_cpu_begin_ns=16,thread_cpu_end_ns=17)
        rows[-1].update(completed_spans=5,truncated=True,monotonic_ns=36)
        rows[-1:-1]=[orphan,dict(event='observation_limit',monotonic_ns=35,
            completed_spans=5,first_unwritten_span=5)]
        original=copy.deepcopy(rows)
        with self.assertRaises(ValueError):summarize_core_step_components(rows)
        result=summarize_core_step_components(rows,allow_partial=True)
        self.assertEqual(rows,original)
        self.assertFalse(result['observation_complete'])
        self.assertTrue(result['truncated'])
        self.assertEqual(result['orphan_span_ids'],[6])
        self.assertEqual(result['last_complete_root_ns'],31)
        self.assertEqual(result['step_count'],1)

    @staticmethod
    def fixture():
        rows = events()
        rows[3]['details']['step_components_v1'] = dict(
            columns=['calls', 'inclusive_wall_ns', 'inclusive_thread_cpu_ns', 'raised'],
            values={'execute_model':[1,12,6,0], 'prepare_inputs':[1,6,3,0]},
            nested_not_additive=True, allocator_before={
                'num_host_alloc':2, 'num_host_free':1, 'host_alloc_time.total':100,
                'host_free_time.total':80},
            allocator_after={'num_host_alloc':4, 'num_host_free':2,
                'host_alloc_time.total':110, 'host_free_time.total':90},
            allocator_scope='process_counters_background_frees_possible',
            allocator_time_unit='microseconds', gpu_kernel_timing=False)
        return rows

    def test_component_nested_times_not_added_or_source_mutated(self):
        from scripts.analyze_control_path_overhead import summarize_core_step_components
        rows=self.fixture(); old=copy.deepcopy(rows)
        result=summarize_core_step_components(rows)
        record,=result['records']
        self.assertEqual(rows,old)
        self.assertEqual(record['step_ns'],20)
        self.assertEqual(record['components']['execute_model'],[1,12,6,0])
        self.assertEqual(record['components']['fullgraph'],[0,0,0,0])
        self.assertEqual(record['allocator_delta']['num_host_alloc'],2)
        # Process background timing may exceed this step's wall duration.
        self.assertEqual(record['allocator_delta']['host_free_time.total'],10)
        self.assertTrue(result['nested_not_additive'])
        self.assertFalse(result['gpu_kernel_timing'])

    def test_missing_unknown_units_nan_negative_and_escaped_times_rejected(self):
        from scripts.analyze_control_path_overhead import summarize_core_step_components
        changes=[
            lambda p:p.update(allocator_time_unit='seconds'),
            lambda p:p.update(gpu_kernel_timing=True),
            lambda p:p['values'].update(unknown=[1,1,1,0]),
            lambda p:p['values'].update(execute_model=[1,21,6,0]),
            lambda p:p['values'].update(execute_model=[1,12,11,0]),
            lambda p:p['values'].update(execute_model=[True,12,6,0]),
            lambda p:p['values'].update(execute_model=[1,12,6,2]),
            lambda p:p['allocator_after'].update(num_host_alloc=1),
            lambda p:p['allocator_after'].update(num_host_alloc=4.0),
            lambda p:p['allocator_after'].update({'host_alloc_time.total':float('nan')}),
            lambda p:p['allocator_before'].pop('num_host_free')]
        for change in changes:
            with self.subTest(change=change):
                rows=self.fixture();change(rows[3]['details']['step_components_v1'])
                with self.assertRaises(ValueError):summarize_core_step_components(rows)
        rows=self.fixture();rows[3]['details'].pop('step_components_v1')
        with self.assertRaises(ValueError):summarize_core_step_components(rows)
        with self.assertRaises(ValueError):summarize_core_step_components(self.fixture()[:-1])

    def test_empty_closed_process_and_crossstep_reset(self):
        from scripts.analyze_control_path_overhead import summarize_core_step_components
        footer=events()[-1];footer['completed_spans']=0
        self.assertEqual(summarize_core_step_components([footer])['step_count'],0)
        rows=self.fixture()
        schedule,step=copy.deepcopy(rows[2:4])
        schedule.update(id=6,parent=5,begin_ns=33,end_ns=36,
                        thread_cpu_begin_ns=16,thread_cpu_end_ns=18)
        step.update(id=5,begin_ns=32,end_ns=52,
                    thread_cpu_begin_ns=16,thread_cpu_end_ns=26)
        rows[-1].update(completed_spans=6,monotonic_ns=53)
        rows[-1:-1]=[schedule,step]
        with self.assertRaisesRegex(ValueError,'reset'):summarize_core_step_components(rows)
        p=step['details']['step_components_v1']
        p['allocator_before']=dict(p['allocator_after'])
        self.assertEqual(summarize_core_step_components(rows)['step_count'],2)


class NativeCoreAnalysis(unittest.TestCase):
    def test_nested_partition_is_exact_and_half_open(self):
        value=summarize_core_host_spans(events())
        self.assertTrue(value['complete'])
        self.assertEqual(value['spans'],4)
        self.assertEqual(core_interval_overlap(value['index']['client'],4,21),2)
        self.assertEqual(core_interval_overlap(value['index']['input_nonclient'],4,21),4)
        self.assertEqual(core_interval_overlap(value['index']['step'],4,21),10)
        self.assertEqual(core_interval_overlap(value['index']['client'],6,10),0)
        self.assertEqual(core_interval_overlap(value['index']['step'],0,100),20)
        self.assertEqual(core_interval_overlap(value['index']['step'],31,31),0)
        self.assertEqual(value['orphan_span_ids'],[])

    def test_corrupted_nesting_duplicates_counts_and_order_reject(self):
        changes=[lambda e:e[0].update(id=1),lambda e:e[0].update(parent=3),
            lambda e:e[0].update(begin_ns=7),lambda e:e[0].update(end_ns=11),
            lambda e:e[2].update(parent=1),lambda e:e[-1].update(completed_spans=3),
            lambda e:e[-1].update(truncated=True),lambda e:e[3].update(begin_ns=9)]
        for change in changes:
            with self.subTest(change=change):
                rows=events();change(rows)
                with self.assertRaises(ValueError):summarize_core_host_spans(rows)

    def test_truncation_or_missing_close_not_filled_as_complete(self):
        rows=events()[:3]
        incomplete=summarize_core_host_spans(rows)
        self.assertFalse(incomplete['complete'])
        self.assertEqual(incomplete['orphan_span_ids'],[4])
        self.assertTrue(incomplete['overhead_is_lower_bound'])
        rows.append(dict(event='observation_limit',monotonic_ns=31,completed_spans=3,first_unwritten_span=3))
        footer=events()[-1]
        footer.update(completed_spans=3,truncated=True)
        rows.append(footer)
        value=summarize_core_host_spans(rows)
        self.assertFalse(value['complete'])
        self.assertTrue(value['closed'])
        self.assertTrue(value['truncated'])

    def test_empty_closed_process_is_retained(self):
        footer=events()[-1]
        footer.update(completed_spans=0)
        value=summarize_core_host_spans([footer])
        self.assertTrue(value['complete'])
        self.assertIsNone(value['first_ns'])
        self.assertEqual(value['metrics'],[])

    def test_input_turns_single_client_has_no_suffix(self):
        original=events();before=copy.deepcopy(original)
        value=summarize_core_host_spans(original,include_input_turns=True)['input_turns']
        self.assertEqual(original,before)
        self.assertEqual(value['input_turns'],1)
        self.assertEqual(value['records'][0]['client_count'],1)
        self.assertEqual(value['records'][0]['utility_count'],1)
        self.assertEqual(value['records'][0]['following_step_begin_ns'],11)
        self.assertTrue(value['records'][0]['following_step_model_executed'])
        self.assertEqual(value['index']['work_clients_after_first']['total_ns'],0)
        self.assertEqual(value['index']['work_utility']['total_ns'],4)

    def test_input_turns_multiple_utilities_and_half_open_clipping(self):
        rows=events()
        rows[0]['details']['request_type']='ADD'
        second=copy.deepcopy(rows[0]);second.update(id=3,begin_ns=6,end_ns=8,
            thread_cpu_begin_ns=3,thread_cpu_end_ns=4)
        second['details']['request_type']='UTILITY'
        third=copy.deepcopy(second);third.update(id=4,begin_ns=8,end_ns=10,
            thread_cpu_begin_ns=4,thread_cpu_end_ns=5)
        rows[2].update(id=6,parent=5);rows[3]['id']=5
        rows[-1]['completed_spans']=6
        rows[1:1]=[second,third]
        value=summarize_core_host_spans(rows,include_input_turns=True)['input_turns']
        turn=value['records'][0]
        self.assertEqual((turn['client_count'],turn['utility_count']),(3,2))
        self.assertEqual((turn['client_ns'],turn['clients_after_first_ns'],
                          turn['clients_after_first_utility_ns']),(8,4,2))
        self.assertEqual(core_interval_overlap(value['index']['work_clients_after_first_utility'],7,9),1)
        self.assertEqual(value['index']['work_utilities_after_first']['total_ns'],2)
        self.assertEqual(value['index']['work_multi_client']['total_ns'],8)
        rows[3]['details']['unfinished_before']=0
        idle=summarize_core_host_spans(rows,include_input_turns=True)['input_turns']
        self.assertEqual(idle['index']['work_client']['total_ns'],0)
        self.assertEqual(idle['records'][0]['client_count'],3)

    def test_input_turns_empty_terminal_and_incomplete(self):
        rows=events()[:2]
        footer=events()[-1];footer['completed_spans']=2
        value=summarize_core_host_spans(rows+[footer],include_input_turns=True)['input_turns']
        self.assertIsNone(value['records'][0]['following_step_begin_ns'])
        with self.assertRaises(ValueError):summarize_core_host_spans(rows,include_input_turns=True)
        footer['completed_spans']=1
        rows=[events()[1],footer];rows[0]['id']=1
        value=summarize_core_host_spans(rows,include_input_turns=True)['input_turns']
        self.assertEqual(value['empty_input_turns'],1)
        self.assertEqual(value['records'],[])

    def write_fixture(self, root):
        directory=root/'diagnostic_core';directory.mkdir()
        receipt=root/'exec_receipt.json';deployment=root/'deployment.json'
        owner='/sys/fs/cgroup/test-unit'
        receipt.write_text(json.dumps(dict(allow_exec=True,service_identity=dict(path=owner),
            external_replay=dict(replay_scope='diagnostic_prefix_v1',diagnostic_prefix_count=1))))
        deployment.write_text(json.dumps(dict(plan=dict(count=1),clock_id='clock')))
        header=dict(event='native_core_host_timing_v1',pid=123,start_ticks=10,cgroup=owner,
            formal_performance_result=False,gpu_kernel_timing=False,observer_overhead_included=True,
            clock_id='clock',diagnostic_prefix_count=1,replay_scope='diagnostic_prefix_v1')
        source=directory/'123-10.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in [header]+events()))
        windows=[dict(request_id='r1',worker_pid=123,replica='replica1',tokens=2,tpot_ms=.000017,
                      timing_tpot_miss=False,first=4e-9,last=21e-9)]
        return directory,receipt,deployment,windows,root/'output',source

    def test_iteration_opt_in_keeps_default_and_source_unchanged(self):
        rows=events();before=copy.deepcopy(rows)
        default=summarize_core_host_spans(rows)
        detailed=summarize_core_host_spans(rows,include_iterations=True)
        iteration=detailed.pop('iterations')
        self.assertEqual(default,detailed)
        self.assertEqual(rows,before)
        row,=iteration['records']
        self.assertEqual(row['category'],'has_multi_token_allocations')
        self.assertEqual((row['step_ns'],row['schedule_ns'],row['step_non_schedule_ns']),(20,3,17))
        self.assertEqual((row['scheduled_requests'],row['scheduled_tokens'],row['multi_token_requests']),(2,11,1))
        self.assertEqual((row['preceding_input_id'],row['preceding_client_count'],row['preceding_utility_count']),(1,1,1))
        self.assertEqual(row['preceding_client_ns'],4)
        index=iteration['index']
        self.assertEqual(core_interval_overlap(index[row['category']],14,21),7)
        self.assertEqual(core_interval_overlap(index['schedule'],14,21),1)
        self.assertEqual(core_interval_overlap(index['step_non_schedule'],14,21),6)
        self.assertFalse(iteration['gpu_kernel_timing'])

    def test_iteration_single_empty_and_absent_schedule_remain_distinct(self):
        rows=events();rows[2]['details'].update(scheduled_tokens=2,multi_token_requests=0)
        self.assertEqual(summarize_core_host_spans(rows,include_iterations=True)
            ['iterations']['records'][0]['category'],'single_token_allocations')
        rows[2]['details'].update(scheduled_requests=0,scheduled_tokens=0)
        rows[3]['details']['model_executed']=False
        self.assertEqual(summarize_core_host_spans(rows,include_iterations=True)
            ['iterations']['records'][0]['category'],'zero_tokens')
        rows.pop(2);rows[2]['id']=3;rows[-1]['completed_spans']=3
        iteration=summarize_core_host_spans(rows,include_iterations=True)['iterations']
        row,=iteration['records']
        self.assertEqual(row['category'],'no_schedule')
        self.assertIsNone(row['scheduled_tokens'])
        self.assertEqual(row['step_non_schedule_ns'],20)

    def test_iteration_rejects_missing_and_impossible_counts(self):
        changes=[lambda e:e[2]['details'].pop('scheduled_requests'),
            lambda e:e[2]['details'].update(scheduled_requests=True),
            lambda e:e[2]['details'].update(multi_token_requests=3),
            lambda e:e[2]['details'].update(scheduled_tokens=2),
            lambda e:e[2]['details'].update(multi_token_requests=0),
            lambda e:e[2]['details'].update(scheduled_requests=0,multi_token_requests=0),
            lambda e:e[3]['details'].pop('model_executed'),
            lambda e:e[3]['details'].update(running_before=-1),
            lambda e:e[3]['details'].update(waiting_before=None)]
        for change in changes:
            with self.subTest(change=change):
                rows=events();change(rows)
                with self.assertRaises(ValueError):summarize_core_host_spans(rows,include_iterations=True)

    def test_iteration_raised_missing_close_and_empty_stream(self):
        rows=events();rows[2].update(outcome='raised',details={})
        rows[3]['outcome']='raised';rows[3]['details'].pop('model_executed')
        row,=summarize_core_host_spans(rows,include_iterations=True)['iterations']['records']
        self.assertEqual(row['category'],'step_raised')
        self.assertIsNone(row['scheduled_tokens'])
        self.assertIsNone(row['model_executed'])
        with self.assertRaises(ValueError):summarize_core_host_spans(rows[:-1],include_iterations=True)
        close=events()[-1];close['completed_spans']=0
        empty=summarize_core_host_spans([close],include_iterations=True)['iterations']
        self.assertEqual(empty['records'],[])
        self.assertTrue(all(v['total_ns']==0 for v in empty['index'].values()))

    def test_iteration_rejects_ambiguous_multiple_schedules(self):
        rows=events();second=copy.deepcopy(rows[2]);second.update(id=5,begin_ns=16,end_ns=18,
            thread_cpu_begin_ns=8,thread_cpu_end_ns=9)
        rows.insert(3,second);rows[-1]['completed_spans']=5
        with self.assertRaisesRegex(ValueError,'at most one schedule'):
            summarize_core_host_spans(rows,include_iterations=True)

    def test_file_binding_output_partition_and_no_overwrite(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            args=self.write_fixture(Path(tmp))
            result=analyze_core_host_timing(*args[:5])
            self.assertEqual(result['complete_request_observations'],1)
            self.assertFalse(result['gpu_kernel_timing'])
            with (args[4]/'decode_windows.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[0]['other_or_unobserved_ms']),.000001)
            self.assertAlmostEqual(sum(float(rows[0][key]) for key in
                ('client_ms','input_nonclient_ms','step_ms','other_or_unobserved_ms')),
                float(rows[0]['decode_ms']))
            with self.assertRaises(FileExistsError):analyze_core_host_timing(*args[:5])

    def test_clock_pid_and_prefix_mismatch_reject(self):
        for key in ('clock','pid','count'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                args=self.write_fixture(Path(tmp))
                if key=='clock':
                    value=json.loads(args[2].read_text());value['clock_id']='other'
                    args[2].write_text(json.dumps(value))
                elif key=='pid': args[3][0]['worker_pid']=456
                else: args[3].append(copy.deepcopy(args[3][0]))
                with self.assertRaises(ValueError):analyze_core_host_timing(*args[:5])
                self.assertFalse(args[4].exists())


if __name__=='__main__':
    unittest.main()
