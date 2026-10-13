"""Lossless BPE JSON schema bridge for HydraServe's pinned tokenizers 0.19.

Tokenizers 0.20 serializes merges as pairs; 0.19 expects space-joined pairs.
Reject ambiguous tokens and unknown shapes. Never catch a load error and use a
different tokenizer. Vocab, ranks, normalization, special tokens and weights
are unchanged. The original artifact remains immutable.
"""
import copy
from hashlib import sha256
import json
from pathlib import Path


def legacy_bpe_json(raw):
    original = json.loads(raw)
    model = original.get('model', {})
    if model.get('type') != 'BPE' or type(model.get('merges')) is not list:
        raise ValueError('only explicit BPE merges schema is supported')
    merges = model['merges']
    if all(isinstance(m, str) for m in merges):
        return raw
    if not all(type(m) is list and len(m) == 2 and
               all(isinstance(t, str) and t and ' ' not in t for t in m)
               for m in merges):
        raise ValueError('ambiguous or mixed BPE merges cannot be converted losslessly')
    converted = copy.deepcopy(original)
    converted['model']['merges'] = [' '.join(m) for m in merges]
    restored = copy.deepcopy(converted)
    restored['model']['merges'] = [m.split(' ') for m in converted['model']['merges']]
    if restored != original:
        raise ValueError('BPE schema conversion changed tokenizer semantics')
    return json.dumps(converted, ensure_ascii=False, separators=(',', ':')).encode()


def materialize_metadata_view(source, destination):
    """Immutable adapter view: symlink all original files, convert ONE JSON.

    Caller must content-verify the original artifact and qualify the resulting
    tokenizer. No tensor reads/copies, download, source deletion or inference.
    The view must be mounted with its original targets in a container.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if not source.is_dir() or source.name != destination.name:
        raise ValueError('adapter view must preserve original adapter identity')
    raw = (source/'tokenizer.json').read_bytes()
    converted = legacy_bpe_json(raw)
    files = sorted(source.iterdir())
    if any(not p.is_file() for p in files):
        raise ValueError('nested or non-file adapter entry is not qualified')
    destination.mkdir(parents=True, exist_ok=False)
    for path in files:
        if path.name == 'tokenizer.json' and converted != raw:
            target = destination/path.name
            with target.open('xb') as f:
                f.write(converted)
            target.chmod(0o444)
        else:
            (destination/path.name).symlink_to(path)
    destination.chmod(0o555)
    return dict(source=str(source), destination=str(destination),
                source_tokenizer_sha256=sha256(raw).hexdigest(),
                delivered_tokenizer_sha256=sha256(converted).hexdigest(),
                schema_changed=converted != raw, copied_weight_bytes=0,
                converted_metadata_bytes=len(converted) if converted != raw else 0,
                linked_files=[p.name for p in files if (destination/p.name).is_symlink()])
