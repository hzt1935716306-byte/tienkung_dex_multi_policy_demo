from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tienkung_demo.voice import ActionManager, infer_text_actions


class FakeWriter:
    def __init__(self) -> None:
        self.calls = []

    def send(self, command, label="", request_id=None, ttl_seconds=None):
        self.calls.append((command, label, request_id, ttl_seconds))
        return request_id


class FakeStatusReader:
    def wait_for_terminal(self, request_id, timeout, poll_seconds=0.05):
        return {"request_id": request_id, "status": "COMPLETED", "reason": ""}


class VoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "command_file": "/tmp/not-used.jsonl",
            "status_file": "/tmp/not-used-status.jsonl",
            "command_ttl_seconds": 5.0,
            "voice": {"action_timeout_seconds": 1.0},
            "motions": {
                "a": {
                    "display_name": "鞠躬",
                    "voice_keywords": ["鞠躬", "鞠个躬", "谢谢"],
                },
                "b": {
                    "display_name": "摆手",
                    "voice_keywords": ["挥手", "挥挥手", "再见"],
                },
            },
        }

    def test_text_mapping_is_whitelisted(self) -> None:
        self.assertEqual(infer_text_actions("请鞠个躬", self.config), ["a"])
        self.assertEqual(infer_text_actions("向我挥挥手", self.config), ["b"])
        self.assertEqual(infer_text_actions("请抬腿", self.config), [])

    def test_action_manager_uses_real_terminal_status(self) -> None:
        writer = FakeWriter()
        manager = ActionManager(self.config, writer, FakeStatusReader())
        try:
            request_id = manager.trigger("a")
            self.assertIsNotNone(request_id)
            result = manager.wait(request_id, timeout=1.0)
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(writer.calls[0][0], "a")
            self.assertEqual(writer.calls[0][2], request_id)
            self.assertEqual(writer.calls[0][3], 5.0)
        finally:
            manager.stop()


if __name__ == "__main__":
    unittest.main()
