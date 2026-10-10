"""Client-observed native-token SSE boundaries, never payload/EOF proxies.

Only output token IDs advance the first/last-token clock. A multi-token event
is one observed delivery event, not evidence of individual kernel token times.
The caller supplies its monotonic receive timestamps, before JSON decoding.
"""
from __future__ import annotations

import codecs
import json
import math


class NativeStreamTiming:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._previous_receive_s: float | None = None
        self.first_s: float | None = None
        self.last_s: float | None = None
        self.token_count = 0
        self.token_events = 0
        self.multi_token_events = 0
        self.done = False

    def feed(self, chunk: str | bytes, received_s: float) -> None:
        if not math.isfinite(received_s):
            raise ValueError("native stream requires a finite monotonic timestamp")
        if self._previous_receive_s is not None and received_s < self._previous_receive_s:
            raise ValueError("native stream receive clock moved backwards")
        self._previous_receive_s = received_s
        text = self._decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        self._buffer = (self._buffer + text).replace("\r\n", "\n")
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            self._event(event, received_s)

    def _event(self, event: str, received_s: float) -> None:
        data = []
        for line in event.splitlines():
            if line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        if not data:
            return  # SSE comments/keepalives are not generated tokens.
        payload = "\n".join(data)
        if payload.strip() == "[DONE]":
            self.done = True
            return
        obj = json.loads(payload)
        if not isinstance(obj, dict):
            raise ValueError("native SSE payload must be an object")
        ids = []
        if isinstance(obj.get("token"), dict):  # S-LoRA native SSE
            if "id" in obj["token"]:
                ids.append(obj["token"]["id"])
        for choice in obj.get("choices", []) or []:  # vLLM OpenAI SSE
            if choice.get("index", 0) != 0:
                raise ValueError("single-completion timing cannot merge choices")
            values = choice.get("token_ids")
            if values is not None:
                if not isinstance(values, list):
                    raise ValueError("native token_ids must be a list")
                ids.extend(values)
        if not ids:
            return  # usage, role, finish and prompt-token-only messages
        if self.done:
            raise ValueError("native token observed after stream DONE")
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid native output token ID")
        if self.first_s is None:
            self.first_s = received_s
        self.last_s = received_s
        self.token_count += len(ids)
        self.token_events += 1
        self.multi_token_events += int(len(ids) > 1)

    def finish(self) -> None:
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer.strip():
            raise ValueError("incomplete native SSE event at response end")

    def metrics(self, response_completed_s: float) -> dict:
        if not math.isfinite(response_completed_s):
            raise ValueError("invalid response completion timestamp")
        if self.last_s is not None and response_completed_s < self.last_s:
            raise ValueError("response completed before its last token")
        return {
            "native_timing_source": "client_native_token_sse_v1",
            "native_first_token_received_s": self.first_s,
            "native_last_token_received_s": self.last_s,
            "native_response_completed_s": response_completed_s,
            "native_timing_token_count": self.token_count,
            "native_timing_token_events": self.token_events,
            "native_timing_multi_token_events": self.multi_token_events,
            "native_response_tail_ms": (
                (response_completed_s - self.last_s) * 1000.0
                if self.last_s is not None else None
            ),
            "native_decode_window_ms": (
                (self.last_s - self.first_s) * 1000.0
                if self.first_s is not None and self.last_s is not None else None
            ),
        }
