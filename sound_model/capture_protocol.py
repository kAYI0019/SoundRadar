"""File-based request/result protocol for rolling captures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import uuid


@dataclass(frozen=True)
class CaptureRequest:
    request_id: str
    requested_at: str
    capture_seconds: float
    provisional_tag: str | None
    trigger: str


@dataclass(frozen=True)
class CaptureResult:
    request_id: str
    success: bool
    audio_path: str | None = None
    metadata_path: str | None = None
    sample_id: str | None = None
    error_code: str | None = None
    message: str | None = None


def new_capture_request(*, capture_seconds: float, provisional_tag: str | None, trigger: str, now=None) -> CaptureRequest:
    now = now or datetime.now()
    short_id = uuid.uuid4().hex[:6]
    return CaptureRequest(
        request_id=f"capture-{now.strftime('%Y%m%d-%H%M%S')}-{short_id}",
        requested_at=now.isoformat(timespec="seconds"),
        capture_seconds=float(capture_seconds),
        provisional_tag=provisional_tag,
        trigger=str(trigger),
    )


def capture_result_path(request_path: str | Path, request_id: str) -> Path:
    path = Path(request_path)
    return path.parent / f"{path.name}.{request_id}.result.json"


def write_capture_request(path: str | Path, request: CaptureRequest) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(request), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def consume_capture_request(path: str | Path) -> CaptureRequest | None:
    path = Path(path).expanduser()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        request = CaptureRequest(**payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return request


def write_capture_result(request_path: str | Path, result: CaptureResult) -> Path:
    path = capture_result_path(request_path, result.request_id)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return path


def consume_capture_result(request_path: str | Path, request_id: str) -> CaptureResult | None:
    path = capture_result_path(request_path, request_id)
    if not path.exists():
        return None
    try:
        result = CaptureResult(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return result
