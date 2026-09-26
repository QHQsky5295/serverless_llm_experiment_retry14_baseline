"""Opt-in observation only, shared unchanged by both Serverless router variants.

The upstream 0.10.2 V1 frontend computes engine-core timestamps but omits them
from RequestOutput. Snapshot those existing scalar stats before the asynchronous
consumer can see later mutations. No scheduling, loading or sampling change.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

OUTPUT_PROCESSOR_SHA256 = '50f5e0aa5d0b7ece086632de6e37376dbeba0a1c0d2f22f95b6759189e76e04d'


def stamp(metrics, name):
    if os.environ.get('SLLM_TC_MEASUREMENT') == '1':
        from faaslora.clock import local_monotonic_clock_id
        clock_id = local_monotonic_clock_id()
        if metrics.get('tc_clock_id', clock_id) != clock_id:
            raise ValueError('cross-host or time-namespace boundary requires clock calibration')
        metrics['tc_clock_id'] = clock_id
        metrics[name] = time.perf_counter()


def snapshot_output(state, output):
    if output is not None:
        if state.stats is None:
            raise ValueError('TC native observation requires log_stats; no timestamp fallback')
        output.metrics = copy.copy(state.stats)
        output.tc_native_lora_name = state.lora_name
    return output


def snapshot_collector(collector, output):
    # RequestOutput.add merges/replaces tokens but does not propagate metrics in
    # 0.10.2. If the producer outruns its consumer, the retained object therefore
    # needs the SAME latest snapshot as its now-updated cumulative token IDs.
    if hasattr(output, 'tc_native_lora_name') and collector.output is not None:
        if collector.output.request_id != output.request_id:
            raise ValueError('native collector mixed request identities')
        collector.output.metrics = copy.copy(output.metrics)
        collector.output.tc_native_lora_name = output.tc_native_lora_name


def install_v1_snapshot():
    import vllm.v1.engine.output_processor as module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != OUTPUT_PROCESSOR_SHA256:
        raise ValueError('unqualified vLLM output-processor version')
    cls = module.RequestState
    old = cls._new_request_output
    if getattr(old, '_ieee_tc_snapshot_v1', False):
        return

    def observed(state, *args, **kwargs):
        return snapshot_output(state, old(state, *args, **kwargs))

    observed._ieee_tc_snapshot_v1 = True
    cls._new_request_output = observed
    old_put = module.RequestOutputCollector.put

    def put(collector, output):
        old_put(collector, output)
        snapshot_collector(collector, output)

    module.RequestOutputCollector.put = put


class NativeRequestObservation:
    def __init__(self, lora_request):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
        self.timeline = NativeV1TokenTimeline(time.perf_counter(), local_monotonic_clock_id())
        self.lora_name = None if lora_request is None else lora_request.lora_name
        self.lora_int_id = None if lora_request is None else lora_request.lora_int_id
        self.prompt_ids = None

    def observe(self, output):
        if (len(output.outputs) != 1
                or getattr(output, 'tc_native_lora_name', object()) != self.lora_name):
            raise ValueError('native output sequence/adapter binding differs')
        ids = output.prompt_token_ids
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('native prompt token IDs missing')
        if self.prompt_ids is not None and list(ids) != self.prompt_ids:
            raise ValueError('native prompt IDs changed during request')
        self.prompt_ids = list(ids)
        if output.finished and output.outputs[0].finish_reason != 'length':
            raise ValueError('fixed-output request did not finish by length')
        self.timeline.observe(output.metrics, output.outputs[0].token_ids, finished=output.finished)

    def finish(self, internal_metrics):
        observed = self.timeline.finalize(time.perf_counter())
        digest = lambda x: hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()
        return dict(**observed, native_prompt_token_ids=list(self.prompt_ids),
                    native_prompt_token_ids_sha256=digest(self.prompt_ids),
                    completion_token_ids=list(self.timeline.token_ids),
                    completion_token_ids_sha256=digest(self.timeline.token_ids),
                    native_lora_name=self.lora_name, native_lora_int_id=self.lora_int_id,
                    native_adapter_binding_source='vllm_v1_engine_request_state',
                    lora_numerical_correctness_qualified=False,
                    control_observation={k: v for k, v in internal_metrics.items()
                                         if k.startswith('tc_') or k in
                                         ('ready_instances_at_enqueue', 'instance_id')})
