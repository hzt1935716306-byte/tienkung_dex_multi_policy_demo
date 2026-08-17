from __future__ import annotations

import json
import time
from pathlib import Path


class CommandWriter:
    def __init__(self, path: str | Path, source: str = "external"):
        self.path = Path(path).expanduser()
        self.source = source
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, command: str, label: str = "") -> None:
        command = command.strip().lower()
        if not command:
            return
        record = {
            "time": time.time(),
            "source": self.source,
            "command": command,
            "label": label,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class CommandReader:
    """Tail a JSONL command file without replaying old commands at startup."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        self.position = 0
        if self.path and self.path.is_file():
            self.position = self.path.stat().st_size

    def read(self) -> list[tuple[str, str]]:
        if self.path is None or not self.path.is_file():
            return []
        if self.path.stat().st_size < self.position:
            self.position = 0

        commands: list[tuple[str, str]] = []
        with self.path.open("r", encoding="utf-8") as file:
            file.seek(self.position)
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    command = str(payload.get("command", payload.get("key", "")))
                    source = str(payload.get("source", "file"))
                except json.JSONDecodeError:
                    command = line
                    source = "text"
                command = command.strip().lower()
                if command:
                    commands.append((command, source))
            self.position = file.tell()
        return commands
