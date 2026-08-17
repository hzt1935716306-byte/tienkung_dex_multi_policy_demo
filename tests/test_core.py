from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tienkung_demo.command_bus import CommandReader, CommandWriter, SshCommandWriter
from tienkung_demo.config import load_config
from tienkung_demo.voice import infer_text_actions


class ConfigTests(unittest.TestCase):
    def test_project_paths_are_resolved(self) -> None:
        config = load_config(ROOT / "configs" / "demo.json")
        self.assertTrue(Path(config["model"]).is_file())
        self.assertTrue(Path(config["walk_policy"]).is_file())
        self.assertEqual(set(config["motions"]), {"a", "b"})

    def test_voice_keyword_mapping(self) -> None:
        config = load_config(ROOT / "configs" / "demo.json")
        self.assertEqual(infer_text_actions("请向大家鞠躬", config), ["a"])
        self.assertEqual(infer_text_actions("欢迎大家，挥挥手", config), ["b"])
        self.assertEqual(infer_text_actions("没有动作", config), [])


class CommandBusTests(unittest.TestCase):
    def test_reader_only_returns_new_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.jsonl"
            path.write_text(json.dumps({"command": "old"}) + "\n", encoding="utf-8")
            reader = CommandReader(path)
            CommandWriter(path, source="test").send("A", "bow")
            self.assertEqual(reader.read(), [("a", "test")])
            self.assertEqual(reader.read(), [])

    @patch("tienkung_demo.command_bus.subprocess.run")
    def test_ssh_writer_forwards_to_remote_send_script(self, run) -> None:
        writer = SshCommandWriter("robot@server", "/srv/tienkung_demo")
        writer.send("A", "bow")
        args = run.call_args.args[0]
        self.assertEqual(args[0:2], ["ssh", "robot@server"])
        self.assertIn("/srv/tienkung_demo/scripts/send_command.py", args[2])
        self.assertIn("--source voice-ssh", args[2])
        run.assert_called_once_with(args, check=True)


if __name__ == "__main__":
    unittest.main()
