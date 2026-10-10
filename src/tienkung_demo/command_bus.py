from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"COMPLETED", "REJECTED", "FAILED", "CANCELLED"}


@dataclass(frozen=True)
class CommandEnvelope:
    request_id: str
    command: str
    source: str
    label: str
    issued_at: float
    expires_at: float | None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


class CommandWriter:
    def __init__(self, path: str | Path, source: str = "external"):
        self.path = Path(path).expanduser()
        self.source = source
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        command: str,
        label: str = "",
        request_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        command = command.strip().lower()
        if not command:
            raise ValueError("Command cannot be empty")
        issued_at = time.time()
        envelope = CommandEnvelope(
            request_id=request_id or uuid.uuid4().hex,
            command=command,
            source=self.source,
            label=label,
            issued_at=issued_at,
            expires_at=(
                None
                if ttl_seconds is None
                else issued_at + max(0.0, float(ttl_seconds))
            ),
        )
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(envelope), ensure_ascii=False) + "\n")
        return envelope.request_id


class SshCommandWriter:
    """Forward commands to a simulator running on another machine over SSH."""

    def __init__(
        self,
        target: str,
        remote_project: str,
        remote_python: str = "python3",
        remote_config: str = "configs/demo.json",
        source: str = "voice-ssh",
    ):
        if not target.strip():
            raise ValueError("SSH target cannot be empty.")
        if not remote_project.startswith("/"):
            raise ValueError("Remote project path must be absolute.")
        self.target = target.strip()
        self.remote_project = remote_project.rstrip("/")
        self.remote_python = remote_python
        self.remote_config = remote_config
        self.source = source

    def send(
        self,
        command: str,
        label: str = "",
        request_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        command = command.strip().lower()
        if not command:
            raise ValueError("Command cannot be empty")
        request_id = request_id or uuid.uuid4().hex
        script = posixpath.join(self.remote_project, "scripts", "send_command.py")
        config = (
            self.remote_config
            if self.remote_config.startswith("/")
            else posixpath.join(self.remote_project, self.remote_config)
        )
        remote_command = shlex.join(
            [
                self.remote_python,
                script,
                command,
                "--config",
                config,
                "--source",
                self.source,
                "--label",
                label,
                "--request-id",
                request_id,
            ]
        )
        if ttl_seconds is not None:
            remote_command += " " + shlex.join(["--ttl", str(float(ttl_seconds))])
        subprocess.run(["ssh", self.target, remote_command], check=True)
        return request_id


class SshStatusReader:
    """Wait for a remote controller status through one non-interactive SSH call."""

    def __init__(
        self,
        target: str,
        remote_project: str,
        remote_python: str = "python3",
        remote_config: str = "configs/multi_evt2.json",
    ):
        self.target = target.strip()
        self.remote_project = remote_project.rstrip("/")
        self.remote_python = remote_python
        self.remote_config = remote_config

    def wait_for_terminal(
        self,
        request_id: str,
        timeout: float,
        poll_seconds: float = 0.05,
    ) -> dict[str, Any] | None:
        script = posixpath.join(self.remote_project, "scripts", "wait_status.py")
        config = (
            self.remote_config
            if self.remote_config.startswith("/")
            else posixpath.join(self.remote_project, self.remote_config)
        )
        remote_command = shlex.join(
            [
                self.remote_python,
                script,
                "--config",
                config,
                "--request-id",
                request_id,
                "--timeout",
                str(float(timeout)),
                "--poll",
                str(float(poll_seconds)),
            ]
        )
        completed = subprocess.run(
            ["ssh", self.target, remote_command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout + 5.0,
        )
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None


class CommandReader:
    """Tail a JSONL command file without replaying old commands at startup."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        self.position = 0
        if self.path and self.path.is_file():
            self.position = self.path.stat().st_size

    def read_records(self) -> list[CommandEnvelope]:
        if self.path is None or not self.path.is_file():
            return []
        if self.path.stat().st_size < self.position:
            self.position = 0

        commands: list[CommandEnvelope] = []
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
                    issued_at = float(payload.get("issued_at", payload.get("time", time.time())))
                    expires_at_value = payload.get("expires_at")
                    expires_at = (
                        None if expires_at_value is None else float(expires_at_value)
                    )
                    request_id = str(payload.get("request_id", uuid.uuid4().hex))
                    label = str(payload.get("label", ""))
                except json.JSONDecodeError:
                    command = line
                    source = "text"
                    issued_at = time.time()
                    expires_at = None
                    request_id = uuid.uuid4().hex
                    label = ""
                command = command.strip().lower()
                if command:
                    commands.append(
                        CommandEnvelope(
                            request_id=request_id,
                            command=command,
                            source=source,
                            label=label,
                            issued_at=issued_at,
                            expires_at=expires_at,
                        )
                    )
            self.position = file.tell()
        return commands

    def read(self) -> list[tuple[str, str]]:
        return [(record.command, record.source) for record in self.read_records()]


class StatusWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        record = {"time": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class StatusReader:
    """Read controller status events without replaying unrelated old entries."""

    def __init__(self, path: str | Path | None, *, from_start: bool = False):
        self.path = Path(path).expanduser() if path else None
        self.position = 0
        if not from_start and self.path and self.path.is_file():
            self.position = self.path.stat().st_size

    def read(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.is_file():
            return []
        if self.path.stat().st_size < self.position:
            self.position = 0
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            file.seek(self.position)
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
            self.position = file.tell()
        return records

    def wait_for_terminal(
        self,
        request_id: str,
        timeout: float,
        poll_seconds: float = 0.05,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        latest: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            for payload in self.read():
                if str(payload.get("request_id", "")) != request_id:
                    continue
                latest = payload
                if str(payload.get("status", "")) in TERMINAL_STATUSES:
                    return payload
            time.sleep(poll_seconds)
        return latest
