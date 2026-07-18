from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from sound_model.capture_protocol import (
    CaptureResult,
    consume_capture_request,
    consume_capture_result,
    new_capture_request,
    write_capture_request,
    write_capture_result,
)


class CaptureProtocolTests(unittest.TestCase):
    def test_request_and_matching_result_round_trip(self):
        request = new_capture_request(
            capture_seconds=5,
            provisional_tag="gunshot",
            trigger="F8",
            now=datetime(2026, 7, 19, 22, 15, 30),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = Path(tmpdir) / "capture-request.json"
            write_capture_request(request_path, request)
            consumed = consume_capture_request(request_path)
            result = CaptureResult(request_id=request.request_id, success=True, audio_path="/tmp/a.wav")
            write_capture_result(request_path, result)
            consumed_result = consume_capture_result(request_path, request.request_id)

        self.assertEqual(consumed, request)
        self.assertEqual(consumed_result, result)
        self.assertFalse(request_path.exists())


if __name__ == "__main__":
    unittest.main()
