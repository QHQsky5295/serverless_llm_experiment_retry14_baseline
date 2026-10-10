"""No-GPU checks for token-event timing; never treat payload/EOF as a token."""
import json
import unittest

from scripts.native_stream_timing import NativeStreamTiming


def event(ids, **extra):
    return 'data: ' + json.dumps({'choices': [{'index': 0, 'token_ids': ids}], **extra}, ensure_ascii=False) + '\n\n'


class NativeStreamTimingTests(unittest.TestCase):
    def test_keepalive_usage_and_completion_tail_do_not_change_token_times(self):
        timing = NativeStreamTiming()
        timing.feed(': keepalive\n\n', 1)
        timing.feed('data: {"choices":[],"usage":{"completion_tokens":0}}\n\n', 2)
        self.assertIsNone(timing.first_s)
        timing.feed(event([1]), 3)
        timing.feed(event([2, 3]), 4)
        timing.feed('data: {"choices":[{"index":0,"finish_reason":"length"}]}\n\n', 5)
        timing.feed('data: [DONE]\n\n', 6)
        timing.finish()
        result = timing.metrics(7)
        self.assertEqual(result['native_first_token_received_s'], 3)
        self.assertEqual(result['native_last_token_received_s'], 4)
        self.assertEqual(result['native_timing_token_count'], 3)
        self.assertEqual(result['native_timing_token_events'], 2)
        self.assertEqual(result['native_timing_multi_token_events'], 1)
        self.assertEqual(result['native_response_tail_ms'], 3000)
        self.assertEqual(result['native_decode_window_ms'], 1000)

    def test_every_byte_split_and_crlf(self):
        wire = event([7], text='汉').replace('\n', '\r\n').encode()
        for split in range(len(wire)+1):
            timing = NativeStreamTiming()
            timing.feed(wire[:split], 1)
            timing.feed(wire[split:], 2)
            timing.finish()
            self.assertEqual(timing.token_count, 1)
            self.assertEqual(timing.first_s, 1 if split == len(wire) else 2)

    def test_native_slora_and_single_token(self):
        timing = NativeStreamTiming()
        timing.feed('data: {"token":{"id":13,"text":"a"}}\n\n', 2)
        timing.finish()
        self.assertEqual(timing.first_s, timing.last_s)
        self.assertEqual(timing.token_count, 1)
        # Caller must keep TPOT N/A for n=1, not interpret zero window as TPOT.
        self.assertEqual(timing.metrics(3)['native_decode_window_ms'], 0)

    def test_prompt_ids_and_text_are_not_output_ids(self):
        timing = NativeStreamTiming()
        timing.feed('data: {"prompt_token_ids":[1],"choices":[{"index":0,"text":"x"}]}\n\n', 1)
        timing.finish()
        self.assertEqual(timing.token_count, 0)

    def test_invalid_ids_and_multiple_choices_rejected(self):
        for ids in ([True], [-1], [1.5], ['1']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                NativeStreamTiming().feed(event(ids), 1)
        with self.assertRaises(ValueError):
            NativeStreamTiming().feed('data: {"choices":[{"index":1,"token_ids":[1]}]}\n\n', 1)

    def test_clock_and_truncated_event_rejected(self):
        for timestamp in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                NativeStreamTiming().feed(event([1]), timestamp)
        timing = NativeStreamTiming()
        timing.feed(event([1]), 2)
        with self.assertRaises(ValueError):
            timing.feed(event([2]), 1)
        with self.assertRaises(ValueError):
            timing.metrics(1)
        timing = NativeStreamTiming()
        timing.feed('data: {"choices":', 1)
        with self.assertRaises(ValueError):
            timing.finish()

    def test_tokens_after_done_rejected(self):
        timing = NativeStreamTiming()
        timing.feed('data: [DONE]\n\n', 1)
        with self.assertRaises(ValueError):
            timing.feed(event([1]), 2)


if __name__ == '__main__':
    unittest.main()
