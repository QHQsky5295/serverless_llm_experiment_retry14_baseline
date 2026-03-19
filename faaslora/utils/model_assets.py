"""Helpers for sharing base-model auxiliary files with LoRA adapter directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List


_AUX_FILENAMES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tekken.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "spiece.model",
)

_AUX_GLOB_PATTERNS = (
    "tokenizer.model.v*",
    "tokenizer.mm.model.v*",
)


def _iter_aux_sources(model_root: Path) -> Iterable[Path]:
    seen = set()
    for name in _AUX_FILENAMES:
        path = model_root / name
        if path.exists():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path
    for pattern in _AUX_GLOB_PATTERNS:
        for path in sorted(model_root.glob(pattern)):
            if path.exists():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def ensure_adapter_support_files(adapter_dir: Path, model_name_or_path: str) -> List[str]:
    """
    Best-effort copy/symlink tokenizer and config files into an adapter directory.

    vLLM looks for tokenizer assets in the LoRA path and emits a warning when they
    are missing. For local base models we can safely share these files with each
    adapter directory.
    """
    model_root = Path(model_name_or_path).expanduser()
    if not model_root.exists() or not model_root.is_dir():
        return []

    adapter_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    for src in _iter_aux_sources(model_root):
        name = src.name
        dst = adapter_dir / name
        if not src.exists():
            continue

        if dst.exists() or dst.is_symlink():
            # Keep already-correct files/symlinks, but aggressively repair
            # stale/broken symlinks that can appear after moving frozen pools.
            try:
                if dst.is_symlink():
                    target = dst.resolve(strict=False)
                    if target == src.resolve():
                        continue
                    dst.unlink()
                else:
                    continue
            except Exception:
                try:
                    if dst.is_symlink() or dst.is_file():
                        dst.unlink()
                    elif dst.exists():
                        shutil.rmtree(dst)
                except Exception:
                    pass

        try:
            # Absolute symlinks are more robust when frozen adapter pools get
            # archived/restored or moved across sibling directories.
            dst.symlink_to(src.resolve())
        except Exception:
            if dst.exists() or dst.is_symlink():
                try:
                    dst.unlink()
                except Exception:
                    pass
            shutil.copy2(src, dst)
        created.append(name)
    return created
