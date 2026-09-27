"""Exact-ID lifetime bridge for the qualified native vLLM frontend.

No scheduling, batching, tokenization or request-ID policy is replaced. The
native submission is joined on cancellation so a later ADD cannot race past an
abort/retirement receipt. Metadata survives frontend output-map deletion.
"""
import asyncio
import vllm

if vllm.__version__ != '0.30.0':
    raise RuntimeError('IEEE native frontend requires qualified vLLM 0.30.0')

from vllm.v1.engine.async_llm import AsyncLLM
from faaslora.clock import local_monotonic_clock_id
from faaslora.scheduling.resource_coordinator import native_prompt_identity


class IEEENativeAsyncLLM(AsyncLLM):
    def __init__(self, *args, **kwargs):
        self.ieee_submissions = {}
        self.ieee_pending = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _descriptor(request, adapter_int_id):
        return dict(prompt_tokens=len(request.prompt_token_ids),
            prompt_sha256=native_prompt_identity(request.prompt_token_ids),
            output_limit=request.sampling_params.max_tokens, adapter_int_id=adapter_int_id)

    async def _pending_rpc(self, operation, intent_id, *args):
        result = await self.engine_core.call_utility_async(
            'ieee_pending_admission', operation, [intent_id, *args])
        if (not isinstance(result, dict) or result.get('kind') != 'ieee_pending_admission_v1'
                or result.get('intent_id') != intent_id
                or result.get('clock_id') != local_monotonic_clock_id()
                or result.get('physical_kv_reservation') is not False):
            raise RuntimeError('pending admission reply identity/clock mismatch')
        return result

    async def ieee_register_pending(self, intent_id, prompt, params, adapter_int_id):
        if intent_id in self.ieee_pending or intent_id in self.ieee_submissions:
            raise ValueError('pending frontend identity already used or closed')
        row = dict(state='registering', prompt=prompt, adapter_int_id=adapter_int_id)
        self.ieee_pending[intent_id] = row
        async def register():
            # The actual native renderer owns BOS/template/length semantics.
            # There is no token-count hint or text-retokenization fallback.
            request = await self.input_processor.process_inputs_async(intent_id, prompt, params,
                supported_tasks=await self.get_supported_tasks())
            descriptor = self._descriptor(request, adapter_int_id)
            if row['state'] == 'closed':
                raise ValueError('pending registration was withdrawn during preprocessing')
            row['descriptor'] = descriptor
            reply = await self._pending_rpc('register', intent_id, descriptor)
            if reply.get('state') != 'pending':
                raise ValueError('native owner did not register pending demand')
            row['state'] = 'pending'
            return reply
        row['registration'] = asyncio.create_task(register())
        return await asyncio.shield(row['registration'])

    async def ieee_generate_pending(self, intent_id, prompt, params, *, lora_request):
        row = self.ieee_pending[intent_id]
        aid = lora_request.lora_int_id if lora_request is not None else None
        if (row['state'] != 'pending' or row['prompt'] != prompt
                or row['adapter_int_id'] != aid
                or row['descriptor']['output_limit'] != params.max_tokens):
            raise ValueError('generation changed or reused its pending admission')
        row['state'] = 'generating'
        try:
            async for output in self.generate(prompt, params, request_id=intent_id,
                                              lora_request=lora_request):
                yield output
        finally:
            row['state'] = 'generation_closed'

    async def ieee_close_pending(self, intent_id):
        row = self.ieee_pending.get(intent_id)
        if row is None:
            # Revoke before acknowledging: a late register must not resurrect it.
            row = self.ieee_pending[intent_id] = dict(state='closed')
        task = row.get('registration')
        if task is not None:
            try:
                await asyncio.shield(task)
            except Exception:
                pass  # Only the core withdrawal receipt can confirm absence.
        if intent_id in self.ieee_submissions:
            return await self.ieee_retire_request(intent_id, abort=True)
        if row['state'] == 'generating':
            raise RuntimeError('generation has not closed its possible native ADD path')
        row['state'] = 'closed'
        return await self._pending_rpc('withdraw', intent_id)

    async def _add_request(self, request, prompt, parent_req, index, queue):
        external, internal = request.external_req_id, request.request_id
        if (parent_req is not None or index != 0 or not external or not internal
                or external in self.ieee_submissions):
            raise ValueError('IEEE native submission requires one fresh exact request identity')
        async def submit():
            row = self.ieee_pending.get(external)
            if row is not None:
                aid = request.lora_request.lora_int_id if request.lora_request is not None else None
                descriptor = self._descriptor(request, aid)
                if row['state'] != 'generating' or descriptor != row['descriptor']:
                    raise ValueError('native input differs from admitted prompt/limit/adapter')
                result = await self._pending_rpc('bind', external, internal, descriptor)
                if result.get('state') != 'bound' or result.get('native_request_id') != internal:
                    raise ValueError('native handoff identity mismatch')
            return await super(IEEENativeAsyncLLM, self)._add_request(
                request, prompt, parent_req, index, queue)
        task = asyncio.create_task(submit())
        self.ieee_submissions[external] = (internal, task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # No timeout can turn an uncertain ADD into permission to release.
            # A wedged/dead core is handled by the external whole-service owner.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def ieee_retire_request(self, external_request_id, *, abort):
        if type(abort) is not bool or external_request_id not in self.ieee_submissions:
            raise ValueError('unobserved native submission; no retirement permission')
        internal, task = self.ieee_submissions[external_request_id]
        await asyncio.shield(task)
        if abort:
            await self.abort(internal, internal=True)
        result = await self.engine_core.call_utility_async('ieee_request_retirement', internal, abort)
        if (not isinstance(result, dict) or result.get('retired') is not True
                or result.get('kind') != 'ieee_native_request_retirement_v1'
                or result.get('request_id') != internal
                or result.get('clock_id') != local_monotonic_clock_id()):
            raise RuntimeError('invalid native request retirement identity/clock')
        return {**result, 'external_request_id': external_request_id}
