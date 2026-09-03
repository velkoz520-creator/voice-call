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
from enum import Enum

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

# ASR 幻听过滤（2026-09-03）：静音/呼吸/摩擦声被 SenseVoice 脑补成单字碎片
# （"嗯""句号"之类）。命中即整轮丢弃——她随口一声"嗯"本就不该让他接话，
# 与闻序三级打断里"短促附和不打断"同理。只作用于语音路径，打字内容不过滤。
ASR_HALLUCINATION_MULTI = {
    "嗯嗯", "嗯嗯嗯", "啊啊", "句号", "逗号", "问号", "感叹号", "省略号",
    "谢谢观看", "谢谢收看", "谢谢大家", "请不吝点赞", "订阅", "关注我们",
    # 英文幻听词（匹配前已去空格去标点并转小写，故 here 无空格）
    "um", "uh", "hm", "mm", "hmm", "mhm", "huh", "bye", "you",
    "thankyou", "thanksforwatching",
}
MIN_SPEECH_RMS = 150  # int16 满量程 32767；低于此当环境音丢弃（比前端 VAD 门限还低，双保险）


def _pcm_rms(pcm: bytes) -> float:
    """整段 PCM16 的均方根音量（抽样步长 8，60s 音频也在毫秒级算完）。"""
    import array
    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    if not samples:
        return 0.0
    picked = samples[::8]
    return (sum(s * s for s in picked) / len(picked)) ** 0.5


def _is_hallucination(text: str) -> bool:
    """去标点后空串或单字碎片 → 幻听；多字短语再对黑名单核对一遍。"""
    t = re.sub(r"[，。！？、,.!?~～…\s]", "", text or "").lower()
    if not t or len(t) <= 1:
        return True
    return t in ASR_HALLUCINATION_MULTI


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


async def _call_gateway(http: aiohttp.ClientSession, turn: dict, metrics: dict | None = None) -> str:
    """大脑：把转录 POST 给网关语音快车道，消费 OpenAI SSE 聚合为整段回复。
    网关侧负责：人格注入 / 通话缓存 / 意图分流 / 记忆检索。本函数只当传声筒。
    metrics 非 None 时记录 gateway_first_at / gateway_done_at（分阶段指标，M1.5）。"""
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
    first_at = None
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
                if first_at is None:
                    first_at = time.time()
                    if metrics is not None:
                        metrics["gateway_first_at"] = first_at
                parts.append(piece)
    done_at = time.time()
    if metrics is not None:
        metrics["gateway_done_at"] = done_at
    return "".join(parts).strip()


async def request_reply(http: aiohttp.ClientSession, turn: dict, metrics: dict | None = None) -> str:
    """路由：配置了 GATEWAY_URL 走网关快车道；否则兼容原 Adapter 协议。"""
    if GATEWAY_URL:
        async with _adapter_sem:
            return await _call_gateway(http, turn, metrics)

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
async def synthesize(http: aiohttp.ClientSession, text: str, metrics: dict | None = None) -> bytes | None:
    """TTS：主赛道 elevenlabs；minimax 为 M1 后桩位。
    没有 TTS provider 时仍回传文本（字幕先行）。
    metrics 非 None 时记录 tts_request_at / tts_first_byte_at / tts_done_at（M1.5）。"""
    if TTS_PROVIDER == "mock" or not text:
        return None
    if metrics is not None:
        metrics["tts_request_at"] = time.time()
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
            first = True
            chunks: list[bytes] = []
            async for chunk in response.content.iter_chunked(16384):
                if first and metrics is not None:
                    metrics["tts_first_byte_at"] = time.time()
                    first = False
                chunks.append(chunk)
            done = time.time()
            if metrics is not None:
                metrics["tts_done_at"] = done
            audio = b"".join(chunks)
            return audio if audio else None
    if TTS_PROVIDER == "minimax":
        raise RuntimeError("minimax TTS 桩位：M1 后接（t2a_v2，需 GROUP_ID）")
    raise RuntimeError("TTS provider is not configured")


async def archive_call(http: aiohttp.ClientSession, call_id: str, turns: list, duration_ms: int) -> bool:
    """挂断归档：通话全文回传网关 /v1/voice/archive（C3：K 自己写摘要进记忆）。
    返回 True=归档成功（或未配置归档端点）；False=失败——失败时调用方不得清空 turns。"""
    if not ARCHIVE_URL:
        return True  # 未配置归档视为已处理（本地开发/mock）
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
                return False
            return True
    except Exception as e:
        print(f"[archive] error: {e}", flush=True)
        return False


@dataclass
class _TurnWrite:
    turn_seq: int
    turn_id: str
    role: str
    text: str
    attempts: int = 0


async def _write_turn_row(http: aiohttp.ClientSession, call_id: str, item: _TurnWrite) -> bool:
    """单条落盘（Supabase upsert）。返回 True=成功（或未配置落盘）。"""
    if not (SB_URL and SB_KEY):
        return True  # 未配置落盘视为已处理（本地开发/mock）
    try:
        async with http.post(SB_URL.rstrip("/") + "/rest/v1/voice_call_turns", json={
            "call_session_id": call_id, "turn_id": item.turn_id, "turn_seq": item.turn_seq,
            "role": item.role, "content": item.text[:4000],
        }, headers={
            "apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}",
            # merge-duplicates=upsert，依赖 voice_call_turns_idem_uidx（call_session_id,turn_id,role）
            "Prefer": "return=minimal,resolution=merge-duplicates",
        }, timeout=aiohttp.ClientTimeout(total=15)) as response:
            if response.status >= 300:
                print(f"[turn-queue] write failed ({response.status}) seq={item.turn_seq}", flush=True)
                return False
            return True
    except Exception as e:
        print(f"[turn-queue] error seq={item.turn_seq}: {e}", flush=True)
        return False


class TurnWriter:
    """M1.5-2 顺序持久化队列：每 Call 一个串行写 worker。
    - 提交方 put_nowait 立即返回——DB 抖动/宕机绝不阻塞网关调用与 TTS 主流程
    - worker 按 turn_seq 入队顺序逐条写；单条失败指数退避重试（1/2/4s，最多 4 次）后
      放弃并留日志，后续轮次继续（读侧按 turn_seq 排序，乱序到达不破序）
    - 幂等：DB 唯一索引 (call_session_id, turn_id, role) + merge-duplicates，重试不产生重复行
    - close()：哨兵+限时等待排空；超时取消——不活过 ClientSession（0 号热修哲学延续）"""
    MAX_ATTEMPTS = 4
    RETRY_BASE_S = 1.0

    def __init__(self, http: aiohttp.ClientSession, call_id: str):
        self.http = http
        self.call_id = call_id
        self.q: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.stopped = False

    def submit(self, turn_seq: int, turn_id: str, role: str, text: str) -> None:
        if self.stopped or not text:
            return
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._worker())
        self.q.put_nowait(_TurnWrite(turn_seq, turn_id, role, text))

    async def close(self, timeout: float = 8.0) -> None:
        self.stopped = True
        if self.task is None:
            return
        self.q.put_nowait(None)  # 哨兵：worker 排空队列后自然退出
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self.task.cancel()
            try:
                await self.task
            except BaseException:
                pass

    async def _worker(self) -> None:
        while True:
            item: _TurnWrite | None = await self.q.get()
            if item is None:
                return
            while item.attempts < self.MAX_ATTEMPTS:
                item.attempts += 1
                if await _write_turn_row(self.http, self.call_id, item):
                    break
                if item.attempts < self.MAX_ATTEMPTS:
                    await asyncio.sleep(self.RETRY_BASE_S * 2 ** (item.attempts - 1))
            else:
                print(f"[turn-queue] give up: call={self.call_id} "
                      f"seq={item.turn_seq} role={item.role}", flush=True)


async def log_metrics(call: "Call", turn_seq: int, turn_id: str, generation_id: int, metrics: dict) -> None:
    """M1.5 第一步：分阶段延迟指标（纯观测，fire-and-forget）。写 Supabase voice_call_metrics + JSON 日志。"""
    try:
        import datetime
        row = {
            "call_session_id": call.id, "turn_seq": turn_seq, "turn_id": turn_id,
            "generation_id": generation_id,
        }
        # 只对 *_at 时间戳键做 epoch→ISO 转换；其余原样（此前无差别转换把 generation_id 转成了 1970 怪串）
        row.update({
            k: (datetime.datetime.fromtimestamp(v, datetime.timezone.utc).isoformat()
                if v else None) for k, v in metrics.items() if k.endswith("_at")
        })
        print("[metrics] " + json.dumps(row, ensure_ascii=False), flush=True)
        if SB_URL and SB_KEY:
            async with aiohttp.ClientSession() as http:
                await http.post(SB_URL.rstrip("/") + "/rest/v1/voice_call_metrics", json=row, headers={
                    "apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}", "Prefer": "return=minimal",
                }, timeout=aiohttp.ClientTimeout(total=15))
    except Exception as e:
        print(f"[metrics] error: {e}", flush=True)


class CallState(str, Enum):
    """M1.5-3：会话显式状态机（闻序蓝图八态的 M1.5 子集；RECONNECTING 等归 M2 断线续接）。
    状态只描述"此刻谁占着话筒"，接收循环的响应速度与状态无关（生成任务已后台化）。"""
    LISTENING = "listening"            # 通道开放，等她开口
    USER_SPEAKING = "user_speaking"    # 她正在说（收音中）
    K_THINKING = "k_thinking"          # 生成任务在途（ASR/网关/TTS）
    K_SPEAKING = "k_speaking"          # 音频已下发（实际播完由前端播放队列自理）
    ENDING = "ending"                  # 挂断/断线，收尾中


@dataclass
class Call:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    audio: bytearray = field(default_factory=bytearray)
    active: bool = False
    generation: int = 0
    turns: list = field(default_factory=list)          # [(role, text)] 归档用
    turn_seq: int = 0                                  # M1.5：接收本轮时生成（与 generation_id 分开）
    writer: "TurnWriter | None" = None                 # M1.5-2：顺序持久化队列（session() 里创建）
    state: CallState = CallState.LISTENING             # M1.5-3：显式状态机
    started_at: float = field(default_factory=time.time)

    def set_state(self, s: CallState) -> None:
        if s is not self.state:
            print(f"[state] {self.state.value} -> {s.value}", flush=True)
            self.state = s

    def begin_turn(self, preroll: bytes = b"") -> None:
        """开始收音。preroll：客户端预滚缓冲（开口前 ~900ms 音频，M1.5-4 防吞句首），
        截断到最近 1 秒防异常大包——ASR 对头部静音不敏感，多补无害。"""
        self.active = True
        self.audio = bytearray(preroll[-SAMPLE_RATE * 2:])

    def end_turn(self) -> bytes:
        self.active = False
        max_bytes = SAMPLE_RATE * 2 * MAX_TURN_SECONDS
        return bytes(self.audio[-max_bytes:])


async def send(ws, message: dict) -> None:
    await ws.send(json.dumps(message, ensure_ascii=False))


async def _finish_turn(ws, call: Call) -> None:
    """轮次收口：状态回 LISTENING 并通知前端——所有出口统一走，防界面卡在'正在传达'。"""
    call.set_state(CallState.LISTENING)
    try:
        await send(ws, {"type": "state", "mode": "listening"})
    except Exception:
        pass  # ws 已断：状态照常收敛，通知尽力而为


async def answer_turn(ws, call: Call, http: aiohttp.ClientSession, pcm: bytes, supplied_text: str = "") -> None:
    if not pcm and not supplied_text:
        await send(ws, {"type": "nothing_heard"})
        return
    turn_id = uuid.uuid4().hex
    call.turn_seq += 1                     # M1.5：接收本轮时生成（与 generation_id 分开，不混用）
    turn_seq = call.turn_seq
    vad_end_at = time.time()               # 自适应停句触发点（闻序分阶段指标）
    metrics: dict = {"vad_end_at": vad_end_at}
    call.set_state(CallState.K_THINKING)
    await send(ws, {"type": "state", "mode": "thinking"})
    try:
        transcript = supplied_text or await transcribe(http, pcm)
        metrics["asr_done_at"] = time.time()
        if not transcript:
            await send(ws, {"type": "nothing_heard"})
            await _finish_turn(ws, call)
            return
        if not supplied_text:  # 打字内容不过滤；只防语音路径的幻听碎片
            if _pcm_rms(pcm) < MIN_SPEECH_RMS or _is_hallucination(transcript):
                print(f"[vad-filter] dropped as hallucination: {transcript!r}", flush=True)
                await send(ws, {"type": "nothing_heard"})
                await _finish_turn(ws, call)
                return
        # 先存事实（内存归档缓冲 + 顺序队列落盘），再通知可能已离线的前端（闻序遗漏二）
        call.turns.append(("她", transcript))
        call.writer.submit(turn_seq, turn_id, "user", transcript)
        await send(ws, {"type": "transcript", "call_session_id": call.id, "turn_id": turn_id, "text": transcript})
        reply = await request_reply(http, {"call_session_id": call.id, "turn_id": turn_id,
                                           "transcript": transcript}, metrics)
        if not reply:
            await _finish_turn(ws, call)
            return
        call.turns.append(("他", reply))
        call.writer.submit(turn_seq, turn_id, "assistant", reply)  # 闻序遗漏一：同 turn_id 不同 role，与 user 轮成对
        call.generation += 1
        generation = call.generation
        metrics["generation_id"] = generation
        spoken, caption = split_for_tts(reply)   # 引号内朗读段转译方言；字幕用清洗后文本（无协议标记）
        await send(ws, {"type": "reply_text", "generation_id": generation, "turn_id": turn_id,
                        "text": caption})  # 闻序热修：空 caption 不回退原始 reply（防协议标签漏进字幕）
        audio = await synthesize(http, spoken, metrics) if spoken else None
        if audio and generation == call.generation:
            call.set_state(CallState.K_SPEAKING)   # 音频已下发（实际播完由前端自理）
            await send(ws, {"type": "audio", "generation_id": generation, "data": base64.b64encode(audio).decode("ascii")})
            await send(ws, {"type": "audio_sentence_end", "generation_id": generation})
        await send(ws, {"type": "generation_end", "generation_id": generation})
        await _finish_turn(ws, call)
        await log_metrics(call, turn_seq, turn_id, generation, metrics)  # 纯观测写入，失败不影响通话
    except Exception as error:  # do not serialize credentials or provider bodies
        await _finish_turn(ws, call)  # 收口先行（内含 ws 断保护），再尽量通知错误
        try:
            await send(ws, {"type": "error", "error": str(error)})
        except Exception:
            pass


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
    end_reason = ""  # ""=异常断开 | "hangup"=正常挂断——所有出口统一走 finally

    async with aiohttp.ClientSession() as http:
        call.writer = TurnWriter(http, call.id)
        pending_generation: asyncio.Task | None = None  # M1.5-3：在途生成任务（至多一个，新顶旧/打断/挂断均即时取消）
        try:
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
                    preroll = b""
                    p64 = event.get("preroll")   # M1.5-4：客户端预滚缓冲（开口前 ~900ms），防 VAD 确认延迟吞句首
                    if p64:
                        try:
                            preroll = base64.b64decode(p64)
                        except Exception:
                            preroll = b""
                    call.begin_turn(preroll)
                    call.set_state(CallState.USER_SPEAKING)
                elif kind == "speech_end" or kind == "text":
                    # M1.5-3 核心：生成任务后台化，接收循环从此不再被 ASR/网关/TTS 阻塞——
                    # 她的打断、挂断、下一句话全部即时响应（闻序要求 100-200ms，实际毫秒级）。
                    pcm = b"" if kind == "text" else call.end_turn()
                    if pending_generation and not pending_generation.done():
                        pending_generation.cancel()  # 新话顶旧话：被顶轮的事实已入档，生成即止
                    pending_generation = asyncio.create_task(
                        answer_turn(ws, call, http, pcm, str(event.get("text", "")) if kind == "text" else ""))
                elif kind == "interrupt":
                    call.generation += 1                     # 在途音频作废（与前端 _stopPlayback 双保险）
                    if pending_generation and not pending_generation.done():
                        pending_generation.cancel()          # 掐断网关/TTS 链路（闻序三级打断的服务端半边）
                    call.set_state(CallState.LISTENING)
                    await send(ws, {"type": "interrupted"})
                elif kind == "hangup":
                    end_reason = "hangup"  # 清理统一走 finally，不再复制一套时序
                    return
        finally:
            # 统一出口（在 ClientSession 关闭前）：收割生成任务 → 排空顺序队列 → 归档。
            # 覆盖三种离开方式：正常 hangup / keepalive 异常断开 / 处理异常——一套时序不再漂移
            call.set_state(CallState.ENDING)
            if pending_generation:
                pending_generation.cancel()  # 任务可能正拿着 http——必须先收割，不允许活过 ClientSession
                try:
                    await asyncio.wait({pending_generation}, timeout=3)
                except Exception:
                    pass
            await call.writer.close(timeout=8)
            if call.turns:
                try:
                    ok = await archive_call(http, call.id, call.turns,
                                            int((time.time() - call.started_at) * 1000))
                    if ok:
                        call.turns.clear()  # 归档成功才清；失败不清空——当前仅保留到会话销毁（归档重试另行实现）
                except Exception as e:
                    print(f"[archive] error: {e}", flush=True)


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
