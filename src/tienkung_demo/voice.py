from __future__ import annotations

import base64
import os
import queue
import re
import threading
import time
import uuid
from typing import Any

from .command_bus import CommandWriter, StatusReader


class ActionManager:
    def __init__(self, config: dict[str, Any], writer=None, status_reader=None):
        self.config = config
        self.command_file = config.get("command_file", "/tmp/tienkung_dex_commands.jsonl")
        self.writer = writer or CommandWriter(self.command_file, source="voice")
        self.status_reader = status_reader or StatusReader(config.get("status_file"))
        self.action_timeout = float(
            config.get("voice", {}).get("action_timeout_seconds", 30.0)
        )
        self.command_ttl = float(config.get("command_ttl_seconds", 15.0))
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self.results: dict[str, dict[str, Any]] = {}
        self.result_events: dict[str, threading.Event] = {}
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def trigger(self, key: str) -> str | None:
        key = key.lower()
        if key not in self.config["motions"]:
            return None
        request_id = uuid.uuid4().hex
        self.result_events[request_id] = threading.Event()
        try:
            self.queue.put_nowait((key, request_id))
        except queue.Full:
            self.result_events.pop(request_id, None)
            print(f"[ACTION] rejected locally: another voice action is pending ({key})")
            return None
        print(
            f"[ACTION] queued {key}: {self.config['motions'][key]['display_name']} "
            f"request_id={request_id}"
        )
        return request_id

    def wait(self, request_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        event = self.result_events.get(request_id)
        if event is None:
            return None
        if not event.wait(self.action_timeout if timeout is None else timeout):
            return None
        return self.results.get(request_id)

    def _run(self) -> None:
        while self.running:
            try:
                key, request_id = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            motion = self.config["motions"][key]
            try:
                sent_id = self.writer.send(
                    key,
                    motion["display_name"],
                    request_id=request_id,
                    ttl_seconds=self.command_ttl,
                )
                print(
                    f"[ACTION] sent {key}, waiting for controller status "
                    f"request_id={sent_id}"
                )
                result = self.status_reader.wait_for_terminal(
                    sent_id,
                    timeout=self.action_timeout,
                )
                if result is None or result.get("status") not in (
                    "COMPLETED",
                    "REJECTED",
                    "FAILED",
                    "CANCELLED",
                ):
                    print(
                        f"[ACTION TIMEOUT] no terminal controller status for "
                        f"request_id={sent_id}"
                    )
                    self.results[request_id] = {
                        "request_id": request_id,
                        "status": "TIMEOUT",
                    }
                else:
                    self.results[request_id] = result
                    print(
                        f"[ACTION RESULT] request_id={sent_id} "
                        f"status={result['status']} reason={result.get('reason', '')}"
                    )
            except Exception as exc:
                self.results[request_id] = {
                    "request_id": request_id,
                    "status": "FAILED",
                    "reason": str(exc),
                }
                print(f"[ACTION ERROR] failed to send {key}: {exc}")
            finally:
                event = self.result_events.get(request_id)
                if event is not None:
                    event.set()
                self.queue.task_done()

    def stop(self) -> None:
        self.running = False


def infer_text_actions(text: str, config: dict[str, Any]) -> list[str]:
    normalized = text.strip().lower()
    if normalized in config["motions"]:
        return [normalized]
    matches = []
    for key, motion in config["motions"].items():
        if any(keyword.lower() in normalized for keyword in motion.get("voice_keywords", [])):
            matches.append(key)
    return matches


def run_text_demo(config: dict[str, Any], writer=None, status_reader=None) -> None:
    actions = ActionManager(config, writer, status_reader)
    examples = " / ".join(motion["voice_keywords"][0] for motion in config["motions"].values())
    print(f"[TEXT] 输入动作描述（例如：{examples}），或直接输入动作键。输入 q 退出。")
    try:
        while True:
            try:
                text = input("你说> ").strip()
            except EOFError:
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            keys = infer_text_actions(text, config)
            if not keys:
                print("[TEXT] 没有匹配到动作。")
            for key in keys:
                actions.trigger(key)
    except KeyboardInterrupt:
        print()
    finally:
        actions.stop()


def build_instructions(config: dict[str, Any]) -> str:
    action_lines = []
    for key, motion in config["motions"].items():
        tag = key.upper()
        keywords = "、".join(motion.get("voice_keywords", []))
        action_lines.append(f"- [动作{tag}]：{motion['display_name']}，适用于：{keywords}")
    actions = "\n".join(action_lines)
    return f"""你叫小福，是一位温暖、耐心、说话简短的陪伴机器人。
请用中文回答，每次回复控制在20到50字，并使用敬语“您”。

当回复语义适合机器人动作时，在回复最后添加一个动作标记：
{actions}

动作标记必须放在回复末尾。一次最多使用一个动作；不适合时不要添加动作标记。
不要把动作标记朗读或解释给用户。"""


def connect_conversation(conversation, timeout: float) -> None:
    import websocket

    conversation.ws = websocket.WebSocketApp(
        conversation.url,
        header=conversation._get_websocket_header(),
        on_message=conversation._on_message,
        on_error=conversation._on_error,
        on_close=conversation._on_close,
    )
    conversation.thread = threading.Thread(target=conversation.ws.run_forever, daemon=True)
    conversation.thread.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if conversation.ws.sock and conversation.ws.sock.connected:
            conversation.callback.on_open()
            return
        time.sleep(0.1)
    raise TimeoutError(f"DashScope websocket did not connect within {timeout:.1f}s")


def run_voice_demo(
    config: dict[str, Any],
    mic_device: int | None = None,
    writer=None,
    status_reader=None,
) -> None:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError('请先设置环境变量：export DASHSCOPE_API_KEY="你的 DashScope API Key"')

    try:
        import dashscope
        import pyaudio
        from dashscope.audio.qwen_omni import (
            AudioFormat,
            MultiModality,
            OmniRealtimeCallback,
            OmniRealtimeConversation,
        )
    except ImportError as exc:
        raise RuntimeError("语音依赖未安装，请执行 ./setup.sh --voice") from exc

    dashscope.api_key = api_key
    voice_config = config.get("voice", {})
    actions = ActionManager(config, writer, status_reader)

    class Callback(OmniRealtimeCallback):
        def __init__(self, audio):
            self.audio = audio
            self.output = None
            self.connected = False
            self.ready = threading.Event()

        def on_open(self):
            print("[VOICE] DashScope connected")
            self.output = self.audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
            self.connected = True
            self.ready.set()

        def on_event(self, message):
            if not isinstance(message, dict):
                return
            event_type = message.get("type", "")
            if event_type == "response.audio.delta" and self.output:
                payload = message.get("delta", "")
                if payload:
                    self.output.write(base64.b64decode(payload))
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = message.get("transcript", "").strip()
                if transcript:
                    print(f"[YOU] {transcript}")
            elif event_type == "response.audio_transcript.done":
                response = message.get("transcript", "").strip()
                if response:
                    keys = [match.lower() for match in re.findall(r"\[动作([A-Z])\]", response)]
                    clean = re.sub(r"\[动作[A-Z]\]", "", response).strip()
                    print(f"[XIAOFU] {clean}")
                    for key in keys:
                        actions.trigger(key)
            elif "error" in event_type.lower():
                print(f"[VOICE ERROR] {message}")

        def on_close(self, code, message):
            print(f"[VOICE] connection closed: {code} {message or ''}")
            self.connected = False
            if self.output:
                self.output.close()

    audio = pyaudio.PyAudio()
    if mic_device is None:
        print("[VOICE] available input devices:")
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                print(f"  [{index}] {info['name']}")

    callback = Callback(audio)
    conversation = OmniRealtimeConversation(model=voice_config["model"], callback=callback)
    timeout = float(voice_config.get("connect_timeout", 30))
    microphone = None
    try:
        print(f"[VOICE] connecting to {voice_config['model']} ...")
        connect_conversation(conversation, timeout)
        if not callback.ready.wait(timeout=timeout):
            raise TimeoutError("DashScope callback did not become ready")
        conversation.update_session(
            output_modalities=[MultiModality.TEXT, MultiModality.AUDIO],
            voice=voice_config["voice"],
            instructions=build_instructions(config),
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            enable_turn_detection=True,
            turn_detection_type="server_vad",
            turn_detection_threshold=0.5,
            enable_input_audio_transcription=True,
        )
        microphone_args = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "input": True,
            "frames_per_buffer": 1600,
        }
        if mic_device is not None:
            microphone_args["input_device_index"] = mic_device
        microphone = audio.open(**microphone_args)
        print("[VOICE] ready; speak into the microphone, Ctrl+C to stop")
        while callback.connected:
            chunk = microphone.read(1600, exception_on_overflow=False)
            conversation.append_audio(base64.b64encode(chunk).decode())
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[VOICE] stopped by user")
    finally:
        actions.stop()
        if microphone:
            microphone.close()
        try:
            conversation.close()
        except Exception:
            pass
        audio.terminate()
