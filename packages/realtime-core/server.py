#!/usr/bin/env python3
"""PaiVoice realtime core — Jester 魔改版（电话项目，2026-09-01）。

原版：model-neutral 通话核心（PCM16 → ASR → Adapter → TTS → SSE/WS 回传）。
魔改三处：
  1. 耳朵：ASR 新增 siliconflow（SenseVoiceSmall，中文主赛道）
  2. 大脑：Adapter 新增 gateway 模式——把转录组装成 OpenAI 格式 POST 给
     网关语音快车道（/v1/chat/completions + call_session_id + pai-voice UA），
     消费其 SSE 流聚合为整段回复。人格/记忆/通话历史全部由网关侧负责，
     本进程不存对话历史（单一事实源在网关通话缓存）。
  3. 嘴：TTS 保留 elevenlabs（主赛道），minimax 留桩（M1 后接）。
新增：挂断时把通话全文 POST 到网关 /v1/voice/archive 归档（K 自己写摘要）。
隐私不变：真 Key 只在服务端环境变量，浏览器不持有任何供应商密钥。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re

from cleanse import split_for_tts
import time
import uuid
import wave
from dataclasses import dataclass, field

import aiohttp
from websockets.asyncio.server import serve

SAMPLE_RATE = 16_000
HOST = os.getenv("PAIVOICE_HOST", "127.0.0.1")
PORT = int(os.getenv("PAIVOICE_PORT", "8780"))
TOKEN = os.getenv("PAIVOICE_TOKEN", "")
ASR_PROVIDER = os.getenv("PAIVOICE_ASR_PROVIDER", "mock")          # mock | groq | siliconflow
TTS_PROVIDER = os.getenv("PAIVOICE_TTS_PROVIDER", "mock")          # mock | elevenlabs | minimax(桩)
ASR_KEY = os.getenv("PAIVOICE_ASR_API_KEY") or os.getenv("GROQ_API_KEY", "")
TTS_KEY = os.getenv("PAIVOICE_TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY", "")
GROQ_MODEL = os.getenv("PAIVOICE_GROQ_ASR_MODEL", "whisper-large-v3-turbo")
ELEVEN_VOICE = os.getenv("PAIVOICE_ELEVEN_VOICE_ID", "")
ELEVEN_MODEL = os.getenv("PAIVOICE_ELEVEN_MODEL", "eleven_multilingual_v2")  # v3 填 eleven_v3
# v3 专属：stability 三档 Creative(0.0 最有表现力)/Natural(0.5 均衡)/Robust(1.0 最稳)。要 audio tags 表现力选前两档
ELEVEN_STABILITY = os.getenv("PAIVOICE_ELEVEN_STABILITY", "")

# 大脑：网关语音快车道（OpenAI 兼容 + SSE）。UA 必须带 pai-voice，网关靠它分流。
GATEWAY_URL = os.getenv("PAIVOICE_GATEWAY_URL", "")
GATEWAY_TOKEN = os.getenv("PAIVOICE_GATEWAY_TOKEN", "")
CLIENT_UA = "pai-voice/0.1 (jester-build)"
# 兼容模式（Gateway 未配置时的原 Adapter 路线）；GATEWAY_URL 优先
ADAPTER_URL = os.getenv("PAIVOICE_ADAPTER_URL", "")
ADAPTER_TOKEN = os.getenv("PAIVOICE_ADAPTER_TOKEN", "")

# 归档：挂断后通话全文回传网关，K 自己写摘要进记忆（C3 拍板）
ARCHIVE_URL = os.getenv("PAIVOICE_ARCHIVE_URL", "")
# 逐轮实时落盘（Supabase voice_call_turns）：断线/崩溃零丢失——每轮转写与回复即写
SB_URL = os.getenv("PAIVOICE_SB_URL", "")
SB_KEY = os.getenv("PAIVOICE_SB_KEY", "")
MAX_TURN_SECONDS = int(os.getenv("PAIVOICE_MAX_TURN_SECONDS", "60"))

_adapter_sem = asyncio.Semaphore(1)  # 同一时刻只投递一轮，避免两句转录并发进网关


def wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


async def transcribe(http: aiohttp.ClientSession, pcm: bytes) -> str:
    """Return text only. Provider errors are intentionally safe to show."""
    if ASR_PROVIDER == "mock":
        return ""
    if not ASR_KEY:
        raise RuntimeError("ASR provider is not configured")

    form = aiohttp.FormData()
    form.add_field("file", wav(pcm), filename="turn.wav", content_type="audio/wav")
    form.add_field("language", "zh")

    if ASR_PROVIDER == "siliconflow":
        form.add_field("model", os.getenv("PAIVOICE_SILICONFLOW_ASR_MODEL",
                                          "FunAudioLLM/SenseVoiceSmall"))
        url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    elif ASR_PROVIDER == "groq":
        form.add_field("model", GROQ_MODEL)
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
    else:
        raise RuntimeError("ASR provider is not configured")

    async with http.post(url, data=form, headers={"Authorization": f"Bearer {ASR_KEY}"}) as response:
        if response.status != 200:
            raise RuntimeError(f"ASR request failed ({response.status})")
        return str((await response.json()).get("text", "")).strip()


async def _call_gateway(http: aiohttp.ClientSession, turn: dict) -> str:
    """大脑：把转录 POST 给网关语音快车道，消费 OpenAI SSE 聚合为整段回复。
    网关侧负责：人格注入 / 通话缓存 / 意图分流 / 记忆检索。本函数只当传声筒。"""
    headers = {
        "authorization": f"Bearer {GATEWAY_TOKEN}",
        "content-type": "application/json",
        "user-agent": CLIENT_UA,
    }
    payload = {
        "call_session_id": turn["call_session_id"],
        "turn_id": turn["turn_id"],
        "messages": [{"role": "user", "content": turn["transcript"]}],
        "stream": True,
        "max_tokens": 600,
    }
    parts: list[str] = []
    async with http.post(GATEWAY_URL, json=payload, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=120, sock_read=90)) as response:
        if response.status != 200:
            raise RuntimeError(f"Gateway request failed ({response.status})")
        async for raw in response.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                piece = json.loads(data)["choices"][0].get("delta", {}).get("content") or ""
            except Exception:
                continue
            if piece:
                parts.append(piece)
    return "".join(parts).strip()


async def request_reply(http: aiohttp.ClientSession, turn: dict) -> str:
    """路由：配置了 GATEWAY_URL 走网关快车道；否则兼容原 Adapter 协议。"""
    if GATEWAY_URL:
        async with _adapter_sem:
            return await _call_gateway(http, turn)

    if not ADAPTER_URL:
        return f"我听见了：{turn['transcript']}" if turn["transcript"] else "我没有听清楚。"
    headers = {"content-type": "application/json"}
    if ADAPTER_TOKEN:
        headers["authorization"] = f"Bearer {ADAPTER_TOKEN}"
    async with http.post(ADAPTER_URL.rstrip("/") + "/turn", json=turn, headers=headers,
                         timeout=aiohttp.ClientTimeout(total=120)) as response:
        if response.status != 200:
            raise RuntimeError(f"Adapter request failed ({response.status})")
        result = await response.json()
    return str(result.get("reply", "")).strip()


# 语气中间协议 → 各家 TTS 方言映射（K 在措辞里只用中间协议，方言由清洗层转译；
# 换 TTS 厂商只改这里，措辞零改动。语法依据各官方文档，比武时逐家实测校准。）
async def synthesize(http: aiohttp.ClientSession, text: str) -> bytes | None:
    """TTS：主赛道 elevenlabs；minimax 为 M1 后桩位。
    没有 TTS provider 时仍回传文本（字幕先行）。"""
    if TTS_PROVIDER == "mock" or not text:
        return None
    if TTS_PROVIDER == "elevenlabs":
        if not TTS_KEY or not ELEVEN_VOICE:
            raise RuntimeError("TTS provider is not configured")
        headers = {"xi-api-key": TTS_KEY, "accept": "audio/mpeg", "content-type": "application/json"}
        body = {"text": text, "model_id": ELEVEN_MODEL}
        if ELEVEN_MODEL == "eleven_v3" and ELEVEN_STABILITY:
            # v3 只收 stability（不支持 v2 的 similarity/style 等设置）
            body["voice_settings"] = {"stability": float(ELEVEN_STABILITY)}
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream"
        async with http.post(url, headers=headers, json=body) as response:
            if response.status != 200:
                raise RuntimeError(f"TTS request failed ({response.status})")
            return await response.read()
    if TTS_PROVIDER == "minimax":
        raise RuntimeError("minimax TTS 桩位：M1 后接（t2a_v2，需 GROUP_ID）")
    raise RuntimeError("TTS provider is not configured")


async def archive_call(http: aiohttp.ClientSession, call_id: str, turns: list, duration_ms: int) -> None:
    """挂断归档：通话全文回传网关 /v1/voice/archive（C3：K 自己写摘要进记忆）。失败仅记日志。"""
    if not ARCHIVE_URL:
        return
    transcript = "\n".join(f"[{r}] {c}" for r, c in turns)
    try:
        async with http.post(ARCHIVE_URL, json={
            "call_session_id": call_id,
            "transcript": transcript,
            "duration_ms": duration_ms,
        }, headers={
            "authorization": f"Bearer {GATEWAY_TOKEN}",
            "content-type": "application/json",
        }, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                print(f"[archive] failed ({response.status})", flush=True)
    except Exception as e:
        print(f"[archive] error: {e}", flush=True)


async def log_turn(http: aiohttp.ClientSession, call_id: str, turn_id: str, role: str, text: str) -> None:
    """逐轮实时落盘（fire-and-forget）：断线/崩溃零丢失。失败仅打印不阻塞通话。"""
    if not (SB_URL and SB_KEY and text):
        return
    try:
        async with http.post(SB_URL.rstrip("/") + "/rest/v1/voice_call_turns", json={
            "call_session_id": call_id, "turn_id": turn_id, "role": role, "content": text[:4000],
        }, headers={
            "apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Prefer": "return=minimal",
        }, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status >= 300:
                print(f"[turn-log] write failed ({response.status})", flush=True)
    except Exception as e:
        print(f"[turn-log] error: {e}", flush=True)


@dataclass
class Call:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    audio: bytearray = field(default_factory=bytearray)
    active: bool = False
    generation: int = 0
    turns: list = field(default_factory=list)          # [(role, text)] 归档用
    started_at: float = field(default_factory=time.time)

    def begin_turn(self) -> None:
        self.active = True
        self.audio.clear()

    def end_turn(self) -> bytes:
        self.active = False
        max_bytes = SAMPLE_RATE * 2 * MAX_TURN_SECONDS
        return bytes(self.audio[-max_bytes:])


async def send(ws, message: dict) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))


async def answer_turn(ws, call: Call, http: aiohttp.ClientSession, pcm: bytes, supplied_text: str = "") -> None:
    if not pcm and not supplied_text:
        await send(ws, {"type": "nothing_heard"})
        return
    turn_id = uuid.uuid4().hex
    try:
        transcript = supplied_text or await transcribe(http, pcm)
        if not transcript:
            await send(ws, {"type": "nothing_heard"})
            return
        await send(ws, {"type": "transcript", "call_session_id": call.id, "turn_id": turn_id, "text": transcript})
        # user 轮用 create_task：写库延迟不阻塞 LLM 请求（闻序回归①——await 会把 Supabase 最坏 15s 塞进开口链路）
        asyncio.create_task(log_turn(http, call.id, turn_id, "user", transcript))
        reply = await request_reply(http, {"call_session_id": call.id, "turn_id": turn_id,
                                           "transcript": transcript})
        if not reply:
            return
        call.turns.append(("她", transcript))
        call.turns.append(("他", reply))
        # assistant 轮 await：回复已到手，此刻写库延迟不影响开口
        await log_turn(http, call.id, turn_id, "assistant", reply)
        call.generation += 1
        generation = call.generation
        spoken, caption = split_for_tts(reply)   # 引号内朗读段转译方言；字幕用清洗后文本（无协议标记）
        await send(ws, {"type": "reply_text", "generation_id": generation, "turn_id": turn_id, "text": caption or reply})
        audio = await synthesize(http, spoken) if spoken else None
        if audio and generation == call.generation:
            await send(ws, {"type": "audio", "generation_id": generation, "data": base64.b64encode(audio).decode("ascii")})
            await send(ws, {"type": "audio_sentence_end", "generation_id": generation})
        await send(ws, {"type": "generation_end", "generation_id": generation})
    except Exception as error:  # do not serialize credentials or provider bodies
        await send(ws, {"type": "error", "error": str(error)})


async def session(ws) -> None:
    call = Call()
    # 鉴权 token 兼容两种传递：start 消息内 token 字段，或拨号 URL 的 ?token=xxx
    url_token = ""
    try:
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(ws.request.path).query)
        url_token = (q.get("token") or [""])[0]
    except Exception:
        pass
    token_ok = (not TOKEN) or (url_token == TOKEN)

    async with aiohttp.ClientSession() as http:
        async for raw in ws:
            if isinstance(raw, bytes):
                if call.active:
                    call.audio.extend(raw)
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            # 鉴权门：未通过 start 鉴权的连接只接受 start——防止匿名连接直接发
            # text/speech_end 白烧 ASR/LLM/TTS 额度（闻序审查 P0）
            if TOKEN and kind != "start" and not token_ok:
                continue
            if kind == "start":
                if TOKEN and not token_ok and event.get("token") != TOKEN:
                    await send(ws, {"type": "error", "error": "Unauthorized"})
                    return
                token_ok = True
                await send(ws, {"type": "state", "call_session_id": call.id, "mode": "listening"})
            elif kind == "speech_start":
                call.begin_turn()
            elif kind == "speech_end":
                pcm = call.end_turn()
                await send(ws, {"type": "state", "mode": "thinking"})
                await answer_turn(ws, call, http, pcm)
                await send(ws, {"type": "state", "mode": "listening"})
            elif kind == "text":
                await send(ws, {"type": "state", "mode": "thinking"})
                await answer_turn(ws, call, http, b"", str(event.get("text", "")))
                await send(ws, {"type": "state", "mode": "listening"})
            elif kind == "interrupt":
                call.generation += 1
                await send(ws, {"type": "interrupted"})
            elif kind == "hangup":
                await archive_call(http, call.id, call.turns,
                                   int((time.time() - call.started_at) * 1000))
                call.turns.clear()  # 已归档，防 finally 兜底重复
                return

    # 异常断开兜底（keepalive 超时/网络掉线）：async for 抛异常会跳过 hangup 分支，
    # 只要通话有内容就在这里归档——别让断线吞掉一整通电话
    if call.turns:
        try:
            async with aiohttp.ClientSession() as http:
                await archive_call(http, call.id, call.turns,
                                   int((time.time() - call.started_at) * 1000))
        except Exception as e:
            print(f"[archive] fallback error: {e}", flush=True)


async def main() -> None:
    from pathlib import Path

    index_paths = [Path(__file__).parent / "index.html",          # 容器：/app/index.html（Dockerfile COPY）
                   Path(__file__).parent.parent / "web-client" / "index.html"]  # 本地开发
    index_text = next((p.read_text(encoding="utf-8") for p in index_paths if p.exists()),
                      "<h1>voice-call page missing</h1>")
    vc_paths = [Path(__file__).parent / "voice-call.js",
                Path(__file__).parent.parent / "web-client" / "voice-call.js"]
    vc_text = next((p.read_text(encoding="utf-8") for p in vc_paths if p.exists()),
                   "export default {}")
    static_files = {
        "/": ("text/html; charset=utf-8", index_text),
        "/index.html": ("text/html; charset=utf-8", index_text),
        "/voice-call.js": ("application/javascript; charset=utf-8", vc_text),
    }

    def process_request(connection, request):  # 静态托管通话页 + voice-call.js；其余路径走 WS 升级
        from websockets.datastructures import Headers
        from websockets.http11 import Response
        try:
            path = request.path.split("?")[0]
            if path in static_files:
                ctype, body_text = static_files[path]
                body = body_text.encode("utf-8")
                return Response(200, "OK", Headers([
                    ("content-type", ctype),
                    ("content-length", str(len(body))),
                    ("access-control-allow-origin", "*"),
                ]), body)
            if path == "/health":
                return Response(200, "OK", Headers([
                    ("content-type", "text/plain"),
                ]), b"ok")
            return None
        except Exception:
            import traceback
            return Response(500, "ERR", Headers([
                ("content-type", "text/plain; charset=utf-8"),
            ]), traceback.format_exc().encode())

    async with serve(session, HOST, PORT, max_size=None, process_request=process_request,
                     ping_interval=25, ping_timeout=120):  # 手机+VPN 链路抖动大，放宽保活判定
        print(f"PaiVoice listening on ws://{HOST}:{PORT}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
