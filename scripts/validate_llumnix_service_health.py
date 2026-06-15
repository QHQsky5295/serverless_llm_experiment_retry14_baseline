#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


SETUP_PATTERN = re.compile(
    r"Init Llumnix components done, (\d+) instances are ready"
)
MANAGER_PATTERN = re.compile(
    r"global_scheduler\.py:\d+\] num_instances: (\d+)"
)
STARTUP_FATAL_PATTERNS = (
    "Failed to scale up instance",
    "is dead, method_name: scale_up",
    "Kill instance ",
)
RUNTIME_FATAL_PATTERNS = (
    "is dead, method_name:",
    "scale_down instance",
    "Failed to send one way rpc request",
    "Unable to put items into queue",
    "output_forwarder.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-log", required=True)
    parser.add_argument("--expected-instances", required=True, type=int)
    parser.add_argument(
        "--phase", choices=("preflight", "final"), required=True
    )
    parser.add_argument("--runtime-offset", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def build_report(
    text: str,
    expected: int,
    phase: str,
    runtime_offset: int | None = None,
) -> dict:
    if runtime_offset is not None:
        if runtime_offset < 0 or runtime_offset > len(text.encode("utf-8")):
            raise ValueError("runtime offset is outside the service log")
        encoded = text.encode("utf-8")
        startup_text = encoded[:runtime_offset].decode(
            "utf-8", errors="replace"
        )
        runtime_text = encoded[runtime_offset:].decode(
            "utf-8", errors="replace"
        )
        replay_start = runtime_text.find("Client received request")
    else:
        replay_start = text.find("Client received request")
        startup_text = text if replay_start < 0 else text[:replay_start]
        runtime_text = "" if replay_start < 0 else text[replay_start:]
    setup_counts = [int(value) for value in SETUP_PATTERN.findall(startup_text)]
    manager_counts = [
        int(value) for value in MANAGER_PATTERN.findall(startup_text)
    ]
    startup_failures = [
        pattern for pattern in STARTUP_FATAL_PATTERNS if pattern in startup_text
    ]
    runtime_failures = [
        pattern for pattern in RUNTIME_FATAL_PATTERNS if pattern in runtime_text
    ]
    return {
        "schema": "relayserve_llumnix_service_health_v1",
        "phase": phase,
        "expected_instances": expected,
        "setup_ready_instances": setup_counts[-1] if setup_counts else None,
        "manager_registered_instances": (
            manager_counts[-1] if manager_counts else None
        ),
        "startup_failures": startup_failures,
        "runtime_failures": runtime_failures,
        "replay_started": replay_start >= 0,
        "runtime_offset_bytes": runtime_offset,
    }


def classify(report: dict) -> tuple[int, str]:
    expected = report["expected_instances"]
    if report["startup_failures"]:
        return 20, "startup actor failure detected"
    setup_count = report["setup_ready_instances"]
    manager_count = report["manager_registered_instances"]
    if setup_count is not None and setup_count != expected:
        return 21, "setup instance count mismatch"
    if setup_count is None or manager_count is None:
        return 10, "instance registration is incomplete"
    if manager_count != expected:
        return 11, "manager registration is incomplete"
    if report["phase"] == "final":
        if not report["replay_started"]:
            return 22, "final validation has no replay evidence"
        if report["runtime_failures"]:
            return 23, "runtime actor failure detected"
    return 0, "healthy"


def main() -> int:
    args = parse_args()
    text = Path(args.service_log).read_text(
        encoding="utf-8", errors="replace"
    )
    report = build_report(
        text,
        args.expected_instances,
        args.phase,
        args.runtime_offset,
    )
    exit_code, status = classify(report)
    report["status"] = status
    report["exit_code"] = exit_code
    encoded = json.dumps(report, sort_keys=True)
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
