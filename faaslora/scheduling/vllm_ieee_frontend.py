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


class IEEENativeAsyncLLM(AsyncLLM):
    def __init__(self, *args, **kwargs):
        self.ieee_submissions = {}
        super().__init__(*args, **kwargs)

    async def _add_request(self, request, prompt, parent_req, index, queue):
        external, internal = request.external_req_id, request.request_id
        if (parent_req is not None or index != 0 or not external or not internal
                or external in self.ieee_submissions):
            raise ValueError('IEEE native submission requires one fresh exact request identity')
        task = asyncio.create_task(super()._add_request(request, prompt, parent_req, index, queue))
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
