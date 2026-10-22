"""Fresh planning observations carried to the pure worker without a graph copy.

This is a representation boundary, not a validated-state certificate or cache.
The native JSON wire stays unchanged. Only the read-only planning RPC opts in;
full decoding and the existing physical-graph validation precede any plan result.
Pickle here is generated locally for small timing metadata, never network input.
"""
from dataclasses import dataclass, field, replace
import pickle

import msgspec


class _PlanningReply(msgspec.Struct):
    ok: bool
    result: msgspec.Raw = msgspec.Raw(b'{}')
    error: str = 'subprocess_engine_rpc_failed'


class _TimingHeader(msgspec.Struct):
    timing: dict | None = None


@dataclass(frozen=True, slots=True)
class FrozenPlanningSnapshot:
    """Immutable *unvalidated* observation; only decode in the pure planner."""
    payload: bytes = field(repr=False)
    _timing: bytes = field(repr=False)

    def __post_init__(self):
        if type(self.payload) is not bytes or type(self._timing) is not bytes:
            raise TypeError('planning wire requires owned immutable bytes')

    def timing_copy(self):
        return pickle.loads(self._timing)

    def with_timing(self, timing):
        return replace(self, _timing=pickle.dumps(timing, protocol=5))

    def decode(self):
        result = msgspec.json.decode(self.payload)
        if type(result) is not dict:
            raise ValueError('planning observation must be a native object')
        result['timing'] = self.timing_copy()
        return result


def decode_planning_rpc_reply(raw):
    """Read RPC status and timing, not the complete allocation/alias object tree.

Raw fields still require a lexical scan. The small timing header is decoded
separately so existing parent RPC accounting is preserved, rather than lost or
reconstructed later with a new response time. Neither scan certifies the graph.
"""
    reply = msgspec.json.decode(raw, type=_PlanningReply)
    if not reply.ok:
        return dict(ok=False, error=reply.error)
    header = msgspec.json.decode(reply.result, type=_TimingHeader)
    result = FrozenPlanningSnapshot(bytes(reply.result), pickle.dumps(header.timing, protocol=5))
    return dict(ok=True, result=result)
