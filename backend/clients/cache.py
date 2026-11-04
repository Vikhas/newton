"""On-disk cache for model responses, so re-runs are instant and free."""
import hashlib
import json
from typing import Any, Callable

from backend import config


def _canonical(v):
    if isinstance(v, (bytes, bytearray)):
        return {"sha256": hashlib.sha256(v).hexdigest()}
    if isinstance(v, dict):
        return {str(k): _canonical(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canonical(x) for x in v]
    return v


def cached_json(namespace: str, key: dict, fn: Callable[[], Any]) -> Any:
    digest = hashlib.sha256(json.dumps(_canonical(key), sort_keys=True).encode()).hexdigest()
    path = config.CACHE_DIR / "models" / namespace / f"{digest}.json"
    if path.exists():
        return json.loads(path.read_text())
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return value
