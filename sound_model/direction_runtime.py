"""Qt-free buffered and asynchronous direction-event inference runtime."""

from __future__ import annotations

import concurrent.futures
import sys
import time

import numpy as np

from .event_temporal_state import EventTemporalState


DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_INTERVAL_SECONDS = 0.50
DEFAULT_TOP_K = 5
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "auto"
DEFAULT_TEACHER_MODEL = "efficientat-mn20"


def normalize_analysis_timing(window_seconds: float, interval_seconds: float) -> tuple[float, float]:
    """Return positive timing with interval capped to prevent unanalyzed gaps."""

    window_seconds = float(window_seconds)
    interval_seconds = float(interval_seconds)
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive")
    if interval_seconds < 0.0:
        raise ValueError("interval_seconds must be non-negative")
    return window_seconds, min(interval_seconds, window_seconds)


class DirectionEventRuntime:
    def __init__(
        self,
        sample_rate,
        channel_count,
        *,
        window_seconds=DEFAULT_WINDOW_SECONDS,
        interval_seconds=DEFAULT_INTERVAL_SECONDS,
        top_k=DEFAULT_TOP_K,
        device=DEFAULT_DEVICE,
        dtype=DEFAULT_DTYPE,
        attn_implementation=None,
        compile_model=False,
        teacher_model=DEFAULT_TEACHER_MODEL,
        model_id=None,
        channel_map=None,
        executor=None,
        score_fn=None,
        latency_clock=None,
        temporal_state=None,
        warmup=True,
    ):
        self.sample_rate = int(sample_rate)
        self.channel_count = int(channel_count)
        requested_interval_seconds = float(interval_seconds)
        self.window_seconds, self.interval_seconds = normalize_analysis_timing(
            window_seconds,
            requested_interval_seconds,
        )
        self.max_samples = max(1, int(self.sample_rate * self.window_seconds))
        self.requested_interval_seconds = requested_interval_seconds
        self.interval_was_capped = self.interval_seconds != self.requested_interval_seconds
        self.top_k = int(top_k)
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.compile_model = compile_model
        self.teacher_model = teacher_model
        self.model_id = model_id
        self.channel_map = dict(channel_map) if channel_map is not None else None
        self._score_fn = score_fn or self._score_with_ast_teacher
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="soundradar-ast",
        )
        self._latency_clock = latency_clock or time.perf_counter
        self._temporal_state = temporal_state if temporal_state is not None else EventTemporalState()
        self._future = None
        self._future_capture_time = None
        self._teacher = None
        self._warmup_future = None
        self._last_submit_time = -float("inf")
        self._audio = np.zeros((0, self.channel_count), dtype=np.float32)
        self.latest_audio_capture_time = None
        self.latest_latency_ms = None
        self.latest_prediction = None
        self.submitted_inference_count = 0
        self.busy_skip_count = 0
        self.disabled_reason = None
        self.resolved_device = None
        self.resolved_dtype = None
        if warmup and score_fn is None:
            self._warmup_future = self._executor.submit(self._warmup_ast_teacher)
            self.poll_warmup()

    def append_blocks(self, blocks, capture_time=None):
        prepared = []
        for block in blocks:
            audio = np.asarray(block, dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            if audio.ndim != 2 or audio.shape[0] == 0:
                continue
            if audio.shape[1] < self.channel_count:
                padded = np.zeros((audio.shape[0], self.channel_count), dtype=np.float32)
                padded[:, : audio.shape[1]] = audio
                audio = padded
            prepared.append(audio[:, : self.channel_count])
        if prepared:
            self._audio = np.concatenate([self._audio, *prepared], axis=0)[-self.max_samples :]
            if capture_time is not None:
                self.latest_audio_capture_time = float(capture_time)

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        try:
            self.latest_prediction = self._future.result()
            if hasattr(self.latest_prediction, "direction_event_scores"):
                self.latest_prediction = self._temporal_state.apply(self.latest_prediction)
            if self._future_capture_time is not None:
                self.latest_latency_ms = max(
                    0.0,
                    (self._latency_clock() - self._future_capture_time) * 1000.0,
                )
        except Exception as exc:
            self.disabled_reason = str(exc)
            print(f"Direction event teacher disabled: {exc}", file=sys.stderr)
            self.latest_prediction = None
        finally:
            self._future = None
            self._future_capture_time = None
        return self.latest_prediction

    def poll_warmup(self):
        if self._warmup_future is None:
            return True
        if not self._warmup_future.done():
            return False
        try:
            self._warmup_future.result()
        except Exception as exc:
            self.disabled_reason = str(exc)
            print(f"Direction event teacher disabled during warmup: {exc}", file=sys.stderr)
        finally:
            self._warmup_future = None
        return self.disabled_reason is None

    def maybe_submit(self, now):
        prediction = self.poll()
        self.poll_warmup()
        if self.disabled_reason is not None or self._warmup_future is not None:
            return prediction
        if self._future is not None:
            if now - self._last_submit_time >= self.interval_seconds:
                self.busy_skip_count += 1
            return prediction
        if self._audio.shape[0] < self.max_samples:
            return prediction
        if now - self._last_submit_time < self.interval_seconds:
            return prediction
        window = np.array(self._audio, copy=True)
        self._last_submit_time = now
        self._future_capture_time = self.latest_audio_capture_time
        self._future = self._executor.submit(
            self._score_fn,
            window,
            self.sample_rate,
            self.top_k,
            "<live>",
        )
        self.submitted_inference_count += 1
        return self.poll() if self._future.done() else prediction

    def _ensure_ast_teacher(self):
        from sound_model.ast_teacher import create_audio_event_teacher

        if self._teacher is None:
            self._teacher = create_audio_event_teacher(
                self.teacher_model,
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
                compile_model=self.compile_model,
            )
        resolved_device = str(getattr(self._teacher, "device", self.device))
        resolved_dtype = str(getattr(self._teacher, "dtype", self.dtype)).replace("torch.", "")
        if self.resolved_device != resolved_device or self.resolved_dtype != resolved_dtype:
            self.resolved_device = resolved_device
            self.resolved_dtype = resolved_dtype
            print(
                f"Direction event teacher {self.teacher_model} "
                f"device: {resolved_device} dtype: {resolved_dtype}"
            )
        return self._teacher

    def _warmup_ast_teacher(self):
        teacher = self._ensure_ast_teacher()
        warmup = getattr(teacher, "warmup_direction_batch", None)
        if callable(warmup):
            warmup(
                sample_rate=self.sample_rate,
                seconds=self.window_seconds,
                direction_count=7,
            )

    def _score_with_ast_teacher(self, audio, sample_rate, top_k, source_path):
        from sound_model.direction_events import score_direction_events

        teacher = self._ensure_ast_teacher()
        return score_direction_events(
            audio,
            sample_rate,
            teacher,
            top_k=top_k,
            source_path=source_path,
            channel_map=self.channel_map,
        )
