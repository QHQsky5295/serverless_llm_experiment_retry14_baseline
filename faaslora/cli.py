"""Command-line entry points for FaaSLoRA."""

from __future__ import annotations

import argparse
import asyncio
import signal
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faaslora",
        description="FaaSLoRA command-line interface",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="faaslora 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    coordinator = subparsers.add_parser(
        "coordinator",
        help="Start a FaaSLoRA coordinator process",
    )
    coordinator.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the FaaSLoRA config file",
    )
    coordinator.add_argument(
        "--host",
        default=None,
        help="Override the coordinator HTTP bind host",
    )
    coordinator.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the coordinator HTTP bind port",
    )
    coordinator.set_defaults(handler=_handle_coordinator)

    return parser


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    def request_shutdown() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: stop_event.set())


async def _run_coordinator(
    *,
    config_path: str,
    host: Optional[str],
    port: Optional[int],
) -> int:
    from faaslora.coordination.coordinator import Coordinator
    from faaslora.utils.config import Config
    from faaslora.utils.logger import get_logger

    logger = get_logger(__name__)
    config = Config(config_path)
    if host:
        config.set("api.http.host", host)
    if port is not None:
        config.set("api.http.port", port)

    coordinator = Coordinator(config)
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    try:
        logger.info(
            "Starting coordinator CLI entrypoint",
            config_path=config_path,
            host=config.get("api.http.host"),
            port=config.get("api.http.port"),
        )
        await coordinator.start()
        await stop_event.wait()
        return 0
    finally:
        await coordinator.stop()


def _handle_coordinator(args: argparse.Namespace) -> int:
    return asyncio.run(
        _run_coordinator(
            config_path=args.config,
            host=args.host,
            port=args.port,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
