from __future__ import annotations

import unittest

from scripts import export_shared_faaslora_trace as export_trace


class ExportSharedFaasloraTraceTests(unittest.TestCase):
    def test_single_request_load_profile_keeps_undefined_average_printable(self):
        trace = type(
            "Trace",
            (),
            {
                "arrival_time": 0.0,
                "expected_input_tokens": 16,
                "expected_output_tokens": 8,
                "adapter_id": "adapter-1",
            },
        )()
        profile = export_trace._build_load_profile(
            [trace],
            configured_time_scale_factor=8.0,
            effective_time_scale_factor=8.0,
            workload_cfg={},
        )
        self.assertIsNone(profile["avg_rps"])
        self.assertEqual(export_trace._format_optional_float(profile["avg_rps"]), "n/a")


if __name__ == "__main__":
    unittest.main()
