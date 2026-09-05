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

from cleanse import split_for_tts, LineSegmenter, ThinkFilter
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
ASR_PROVIDER = os.getenv("PAIVOICE_ASR_PROVIDER", "mock")          # mock | groq | siliconflow | local
TTS_PROVIDER = os.getenv("PAIVOICE_TTS_PROVIDER", "mock")          # mock | elevenlabs | minimax(桩)
ASR_KEY = os.getenv("PAIVOICE_ASR_API_KEY") or os.getenv("GROQ_API_KEY", "")
TTS_KEY = os.getenv("PAIVOICE_TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY", "")
GROQ_MODEL = os.getenv("PAIVOICE_GROQ_ASR_MODEL", "whisper-large-v3-turbo")
ELEVEN_VOICE = os.getenv("PAIVOICE_ELEVEN_VOICE_ID", "")
ELEVEN_MODEL = os.getenv("PAIVOICE_ELEVEN_MODEL", "eleven_multilingual_v2")  # v3 填 eleven_v3
# v3 专属：stability 三档 Creative(0.0 最有表现力)/Natural(0.5 均衡)/Robust(1.0 最稳)。要 audio tags 表现力选前两档
ELEVEN_STABILITY = os.getenv("PAIVOICE_ELEVEN_STABILITY", "")

# 容器本地 ASR（9-05 天天拍板）：SenseVoiceSmall ONNX 经 sherpa-onnx 跑在容器内——
# 同一只耳朵（比武场冠军同款模型），砍掉跨太平洋上传（9-03 实测 ASR 平均 5.5s 的大头）。
# 模型文件由 Dockerfile 构建期下载（Ashburn→GitHub 快线），路径可用 PAIVOICE_LOCAL_ASR_DIR 覆盖。
LOCAL_ASR_DIR = os.getenv("PAIVOICE_LOCAL_ASR_DIR", "/app/asr-model")
_local_recognizer = None          # 懒加载（首句才初始化，不拖启动）
_local_recognizer_lock = __import__("threading").Lock()
_local_decode_lock = __import__("threading").Lock()   # OfflineRecognizer 串行 decode（保守线程安全）

# VAD 切段（9-05 天天转述朋友方案并拍板）：silero-vad 边说边切，段闭合即后台识别，
# 她说完时通常只剩尾段——ASR 延迟不再随说话时长涨。模型运行时下载（2.3MB，一次）。
ASR_CHUNK = os.getenv("PAIVOICE_ASR_CHUNK", "1") == "1"
SILERO_VAD_MODEL_URL = os.getenv(
    "PAIVOICE_SILERO_VAD_URL",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx")
SENSEVOICE_MODEL_URL = os.getenv(
    "PAIVOICE_SENSEVOICE_URL",
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx")
SENSEVOICE_TOKENS_URL = os.getenv(
    "PAIVOICE_SENSEVOICE_TOKENS_URL",
    "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt")
VAD_MAX_SEGMENT_S = float(os.getenv("PAIVOICE_VAD_MAX_SEGMENT_S", "25"))
VAD_CTX_SAMPLES = int(SAMPLE_RATE * 0.4)   # 段头垫的起音前上下文（防 silero 起音漏判吞字）
_vad_model = None                 # VadModel 单例（状态在实例上，Call 层用 reset 复用）
_vad_model_lock = __import__("threading").Lock()


def _ensure_model_file(path: str, url: str) -> str:
    """模型文件缺失时运行时下载（zbpack 缓存坑的兜底：镜像没带模型也能自愈）。
    用 urllib 标准库——容器里没有 requests（9-05 实测踩坑）。返回路径；失败抛异常由调用方回落。"""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    print(f"[asr-local] downloading {url} -> {path}", flush=True)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    return path

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

# 自然挂断（COVE §16 / M2）：告别词命中 → 正常生成告别回复 → 前端播完（playback_idle）+
# 宽限期她没再开口 → 请前端挂断收线；全程硬截止，超时强制关连接（归档统一走 finally）。
HANGUP_GRACE_MS = int(os.getenv("PAIVOICE_HANGUP_GRACE_MS", "3500"))
HANGUP_DEADLINE_S = float(os.getenv("PAIVOICE_HANGUP_DEADLINE_S", "30"))
FAREWELL_RE = re.compile(
    r"(先挂了|挂了哈|挂了吧|挂断了|那我挂|拜拜|再见|晚安|先睡了|睡了哈|先这样|去忙了|先去忙|上班去了|干活去了)")
# 反例保护："别挂/不许挂"含"挂"字绝不能当告别。
# 注意"不聊了/不说了"故意不收——话题转换也这么说，误挂比漏挂（她手动挂）事故得多
FAREWELL_NEG_RE = re.compile(r"(别挂|不许挂|不准挂|不要挂|不能挂|谁挂|还没挂|没挂)")

# V2 流式 TTS 总开关（B6 三保险精神）：SSE 增量→切句→逐段合成下发，首句不再等全量。
# 出问题 env 置 0 秒回整段老路（代码保留原路径为回落）。
STREAM_TTS = os.getenv("PAIVOICE_STREAM_TTS", "1") == "1"

_adapter_sem = asyncio.Semaphore(1)  # 同一时刻只投递一轮，避免两句转录并发进网关

# ASR 幻听过滤（2026-09-03）：静音/呼吸/摩擦声被 SenseVoice 脑补成单字碎片
# （"嗯""句号"之类）。命中即整轮丢弃——她随口一声"嗯"本就不该让他接话，
# 与闻序三级打断里"短促附和不打断"同理。只作用于语音路径，打字内容不过滤。
ASR_HALLUCINATION_MULTI = {
    "嗯嗯", "嗯嗯嗯", "啊啊", "句号", "逗号", "问号", "感叹号", "省略号",
    "谢谢观看", "谢谢收看", "谢谢大家", "请不吝点赞", "订阅", "关注我们",
    # 英文幻听词（匹配前已去空格去标点并转小写，故 here 无空格）
    # 9-05 她实测补：VAD 漏检碎片段常识别成单虚词（"the"），一并拉黑；
    # 9-05 晚补：阈值降到 0.3 后呼吸段变多，SenseVoice 对气流声的经典幻觉
    # "okay"（拼全词，不是 ok）高频出没——幽灵 okay 案
    "um", "uh", "hm", "mm", "hmm", "mhm", "huh", "bye", "you",
    "thankyou", "thanksforwatching",
    "the", "a", "an", "and", "to", "it", "is", "or", "so", "yeah", "ok",
    "okay", "hello", "hi", "well", "right",
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


def _get_local_recognizer():
    """懒加载容器本地 SenseVoice 识别器（sherpa-onnx + int8 ONNX，纯 CPU）。
    双重检查锁：首句才初始化（不拖容器启动），并发轮只建一次。
    模型文件缺失时运行时下载（一次 ~240MB，Ashburn→HF 快线）。"""
    global _local_recognizer
    if _local_recognizer is not None:
        return _local_recognizer
    with _local_recognizer_lock:
        if _local_recognizer is not None:
            return _local_recognizer
        import sherpa_onnx
        model_path = _ensure_model_file(os.path.join(LOCAL_ASR_DIR, "model.int8.onnx"), SENSEVOICE_MODEL_URL)
        tokens_path = _ensure_model_file(os.path.join(LOCAL_ASR_DIR, "tokens.txt"), SENSEVOICE_TOKENS_URL)
        t0 = time.time()
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path, tokens=tokens_path, use_itn=True,
            num_threads=max(1, (os.cpu_count() or 2) // 2),
        )
        print(f"[asr-local] SenseVoice int8 loaded in {time.time() - t0:.1f}s "
              f"(threads={max(1, (os.cpu_count() or 2) // 2)})", flush=True)
        _local_recognizer = recognizer
        return recognizer


def _get_vad_model():
    """Silero VAD 单例（模型 2.3MB，缺失时运行时下载）。"""
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    with _vad_model_lock:
        if _vad_model is not None:
            return _vad_model
        import sherpa_onnx
        model_path = _ensure_model_file(os.path.join(LOCAL_ASR_DIR, "silero_vad.onnx"), SILERO_VAD_MODEL_URL)
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        # 9-05 她实测调参：0.5 对轻声尾音太严（"我就先修改呗"整个尾段被判静音丢弃）；
        # 0.3 + 0.7s 闭合更保守——尾音并入段里，代价是段稍长
        cfg.silero_vad.threshold = 0.3
        cfg.silero_vad.min_silence_duration = 0.7
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.sample_rate = SAMPLE_RATE
        _vad_model = sherpa_onnx.VadModel.create(cfg)
        print("[asr-vad] silero VAD ready", flush=True)
        return _vad_model


def _transcribe_local(pcm: bytes) -> str:
    """同步识别（sherpa-onnx 无原生异步）；直接吃 PCM16，连 WAV 编码都省了。
    在 to_thread 里跑，不碰事件循环。decode 全局锁串行（保守线程安全）。"""
    import array
    recognizer = _get_local_recognizer()
    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples.tolist())
    with _local_decode_lock:
        recognizer.decode_stream(stream)
        return stream.result.text.strip()


async def transcribe(http: aiohttp.ClientSession, pcm: bytes) -> str:
    """Return text only. Provider errors are intentionally safe to show.
    local 容器内推理；异常时自动回落 SiliconFlow（同一只耳朵的云备份），电话不断。"""
    if ASR_PROVIDER == "mock":
        return ""
    if ASR_PROVIDER == "local":
        try:
            t0 = time.time()
            text = await asyncio.to_thread(_transcribe_local, pcm)
            print(f"[asr-local] {len(pcm) // 3200}0ms audio -> {time.time() - t0:.2f}s "
                  f"({(len(pcm) // 2) / SAMPLE_RATE:.1f}s audio)", flush=True)
            return text
        except Exception as e:
            if not ASR_KEY:
                raise
            print(f"[asr-local] failed, falling back to siliconflow: {e}", flush=True)
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


async def _call_gateway(http: aiohttp.ClientSession, turn: dict, metrics: dict | None = None,
                        on_segment=None) -> str:
    """大脑：把转录 POST 给网关语音快车道，消费 OpenAI SSE 聚合为整段回复。
    网关侧负责：人格注入 / 通话缓存 / 意图分流 / 记忆检索。本函数只当传声筒。
    metrics 非 None 时记录 gateway_first_at / gateway_done_at（分阶段指标，M1.5）。
    on_segment 非 None（V2 流式）：SSE 增量喂切句器，每切出一段就 await 回调——
    首句不等全量（COVE §12）；返回值仍是完整整段（归档/落盘语义不变）。"""
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
    segmenter = LineSegmenter() if on_segment is not None else None
    think_filter = ThinkFilter()   # thinking 渠道把推理混进 content（9-05 朗读内心独白案）

    async def _emit(line: str) -> None:
        if on_segment is not None:
            await on_segment(line)

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
                clean = think_filter.feed(piece)   # <think> 块剥在切句/归档之前：字幕朗读落盘三处干净
                if not clean:
                    continue
                parts.append(clean)
                if segmenter is not None:
                    for seg in segmenter.feed(clean):
                        await _emit(seg)
        if think_filter.in_think:
            print("[gateway] warn: <think> 未闭合到流尾，推理块已整体丢弃", flush=True)
        tail_text = think_filter.flush()
        if tail_text:
            parts.append(tail_text)
            if segmenter is not None:
                for seg in segmenter.feed(tail_text):
                    await _emit(seg)
        if segmenter is not None:
            tail = segmenter.flush()
            if tail:
                await _emit(tail)
    done_at = time.time()
    if metrics is not None:
        metrics["gateway_done_at"] = done_at
    return "".join(parts).strip()


async def request_reply(http: aiohttp.ClientSession, turn: dict, metrics: dict | None = None,
                        on_segment=None) -> str:
    """路由：配置了 GATEWAY_URL 走网关快车道；否则兼容原 Adapter 协议。
    on_segment 仅网关路径支持（V2 流式）；Adapter/兜底路径忽略之。"""
    if GATEWAY_URL:
        async with _adapter_sem:
            return await _call_gateway(http, turn, metrics, on_segment=on_segment)

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
# MiniMax 备胎 TTS（9-05 天天拍板方案）：主赛道 ElevenLabs 断气（额度/风控）时
# 同轮自动切 MiniMax t2a_v2——电话永不断气。音色需在 MiniMax 侧单独挑（env 配置）。
MINIMAX_API_KEY = os.getenv("PAIVOICE_MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("PAIVOICE_MINIMAX_GROUP_ID", "")
MINIMAX_VOICE_ID = os.getenv("PAIVOICE_MINIMAX_VOICE_ID", "")
MINIMAX_MODEL = os.getenv("PAIVOICE_MINIMAX_MODEL", "speech-01-turbo")


def _minimax_ready() -> bool:
    return bool(MINIMAX_API_KEY and MINIMAX_GROUP_ID and MINIMAX_VOICE_ID)


async def _synthesize_with(http: aiohttp.ClientSession, provider: str, text: str,
                           metrics: dict | None = None) -> bytes | None:
    if provider == "elevenlabs":
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
                # 带响应体：401 风暴（9-05 她实测，持续 19 分钟后自愈）到底是 quota_exceeded
                # 还是限流，不看 body 只能瞎猜
                detail = (await response.text())[:200]
                raise RuntimeError(f"TTS request failed ({response.status}): {detail}")
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
    if provider == "minimax":
        if not _minimax_ready():
            raise RuntimeError("minimax TTS not configured (need API_KEY/GROUP_ID/VOICE_ID)")
        # t2a_v2：响应 JSON 的 data.audio 是 hex 编码 mp3（MiniMax 特色）
        url = f"https://api.minimax.chat/v1/t2a_v2?GroupId={MINIMAX_GROUP_ID}"
        headers = {"authorization": f"Bearer {MINIMAX_API_KEY}", "content-type": "application/json"}
        body = {
            "model": MINIMAX_MODEL,
            "text": text,
            "stream": False,
            "voice_setting": {"voice_id": MINIMAX_VOICE_ID, "speed": 1.0, "vol": 1.0, "pitch": 0},
            "audio_setting": {
                "sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1,
            },
        }
        async with http.post(url, headers=headers, json=body) as response:
            if response.status != 200:
                detail = (await response.text())[:200]
                raise RuntimeError(f"minimax TTS failed ({response.status}): {detail}")
            data = await response.json()
            # MiniMax 把业务错误包在 200 里：base_resp.status_code 非 0 = 失败
            base = data.get("base_resp") or {}
            if base.get("status_code", 0) != 0:
                raise RuntimeError(f"minimax TTS biz error {base.get('status_code')}: {base.get('status_msg', '')[:120]}")
            audio_hex = (data.get("data") or {}).get("audio", "")
            if not audio_hex:
                return None
            audio = bytes.fromhex(audio_hex)
            if metrics is not None and metrics.get("tts_first_byte_at") is None:
                metrics["tts_first_byte_at"] = time.time()
                metrics["tts_done_at"] = time.time()
            return audio or None
    raise RuntimeError("TTS provider is not configured")


async def synthesize(http: aiohttp.ClientSession, text: str, metrics: dict | None = None) -> bytes | None:
    """TTS：主赛道 elevenlabs；断气（额度/风控/网络）时自动切 MiniMax 备胎（9-05 拍板：
    '主赛道 ElevenLabs，额度烧干时自动切过去，电话永不断气'）。备胎未配置则原样抛错。
    没有 TTS provider 时仍回传文本（字幕先行）。
    metrics 非 None 时记录 tts_request_at / tts_first_byte_at / tts_done_at（M1.5）。"""
    if TTS_PROVIDER == "mock" or not text:
        return None
    if metrics is not None:
        metrics["tts_request_at"] = time.time()
    try:
        return await _synthesize_with(http, TTS_PROVIDER, text, metrics)
    except Exception as primary_err:
        # 备胎只在主赛道是 elevenlabs 且其失败时接管；minimax 自身失败不再套娃
        if TTS_PROVIDER == "elevenlabs" and _minimax_ready():
            print(f"[tts] primary failed, falling back to minimax: {str(primary_err)[:150]}", flush=True)
            try:
                audio = await _synthesize_with(http, "minimax", text, None)
                if metrics is not None:
                    metrics["tts_fallback"] = "minimax"
                return audio
            except Exception as backup_err:
                print(f"[tts] minimax backup also failed: {str(backup_err)[:150]}", flush=True)
                raise backup_err   # 双双阵亡必须上抛——静默 None 会让熔断器永不计数（自测抓到）
        raise


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
    pending_generation: object = None                  # M1.5-3：在途生成任务（新轮确认有话时才顶替它）
    state: CallState = CallState.LISTENING             # M1.5-3：显式状态机
    started_at: float = field(default_factory=time.time)
    # 自然挂断（COVE §16 / M2）：告别轮标记 + 收线任务 + 前端播放排空时刻
    farewell: bool = False
    hangup_task: "asyncio.Task | None" = None
    playback_idle_at: float = 0.0
    # VAD 切段（朋友方案）：窗口状态 + 已闭合段的后台转写任务
    vad: object = None                  # 本通话的 VadModel（begin_turn 时 reset 复用）
    vad_cur: list = field(default_factory=list)          # 当前段累积样本
    vad_speech_win: int = 0             # 当前段有声样本数
    vad_silence_win: int = 0            # 当前段尾部静音样本数
    vad_tail: list = field(default_factory=list)         # 滚动音频尾（起音前上下文，0.4s）
    vad_seg_ctx: list = field(default_factory=list)      # 段起音时刻的上下文快照（垫进段头）
    seg_tasks: list = field(default_factory=list)        # 在途段转写 asyncio.Task

    def set_state(self, s: CallState) -> None:
        if s is not self.state:
            print(f"[state] {self.state.value} -> {s.value}", flush=True)
            self.state = s

    def begin_turn(self, preroll: bytes = b"") -> None:
        """开始收音。preroll：客户端预滚缓冲（开口前 ~900ms 音频，M1.5-4 防吞句首），
        截断到最近 1 秒防异常大包——ASR 对头部静音不敏感，多补无害。"""
        self.active = True
        self.audio = bytearray(preroll[-SAMPLE_RATE * 2:])
        # VAD 切段：新一轮清状态（在途段任务引用换新，旧任务自然完成结果弃用）
        self.vad_cur = []
        self.vad_speech_win = 0
        self.vad_silence_win = 0
        self.vad_tail = []
        self.vad_seg_ctx = []
        self.seg_tasks = []
        if ASR_CHUNK and ASR_PROVIDER == "local":
            # 切段的前提是本地识别（段识别写死 _transcribe_local）；provider 切回
            # 云端时切段自动解除，绝不意外加载 240MB 本地模型
            try:
                if self.vad is None:
                    self.vad = _get_vad_model()
                else:
                    self.vad.reset()
            except Exception as e:
                print(f"[asr-vad] unavailable, whole-turn fallback: {e}", flush=True)
                self.vad = None
                return
            # 9-05 她实测"句首时而吞字"：预滚（开口前0.9s）必须喂给 silero，
            # 否则起音在实时流开始前就丢了——句首补料
            if preroll:
                _vad_feed(self, bytes(preroll))
        else:
            self.vad = None

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


def _vad_feed(call: Call, pcm: bytes) -> None:
    """喂 VAD 窗口；段闭合即创建后台转写任务。同步轻量（Silero ~0.001 实时率）。
    在 WS 接收循环里跑，不阻塞收流。"""
    if not ASR_CHUNK or call.vad is None:
        return
    import array
    samples = array.array("h")
    samples.frombytes(pcm[: (len(pcm) // 2) * 2])
    win = call.vad.window_size()
    min_sil = call.vad.min_silence_duration_samples()
    min_sp = call.vad.min_speech_duration_samples()
    max_sp = int(SAMPLE_RATE * VAD_MAX_SEGMENT_S)
    for pos in range(0, len(samples) - win + 1, win):
        window = samples[pos:pos + win]
        hot = call.vad.is_speech(window)
        if hot and not call.vad_cur:
            # 起音时刻：快照此前 0.4s 原始音频垫进段头——silero 刚 reset 判定偏晚时，
            # 起音真音频照样进识别料（9-05 她实测开口吞字案）
            call.vad_seg_ctx = list(call.vad_tail)
        if hot:
            call.vad_cur.extend(window)
            call.vad_speech_win += win
            call.vad_silence_win = 0
            if call.vad_speech_win >= max_sp:          # 单段硬上限强切
                _schedule_segment(call, call.vad_cur, ctx=None)   # 段中强切无起音问题，不垫
                call.vad_cur, call.vad_speech_win, call.vad_silence_win = [], 0, 0
        else:
            if call.vad_cur:
                call.vad_cur.extend(window)
                call.vad_silence_win += win
                if call.vad_silence_win >= min_sil:    # 静音过阈：段闭合
                    if call.vad_speech_win >= min_sp:
                        _schedule_segment(call, call.vad_cur[:len(call.vad_cur) - call.vad_silence_win],
                                          ctx=call.vad_seg_ctx)
                    call.vad_cur, call.vad_speech_win, call.vad_silence_win = [], 0, 0
        call.vad_tail.extend(window)                    # 滚动音频尾（含静音窗）
        if len(call.vad_tail) > VAD_CTX_SAMPLES:
            del call.vad_tail[:len(call.vad_tail) - VAD_CTX_SAMPLES]


def _schedule_segment(call: Call, samples_list: list, ctx: list | None = None) -> None:
    """闭合段 → 后台转写任务（结果挂在任务上，speech_end 时统一收割拼接）。
    ctx：起音前上下文（垫段头，防起音漏判吞字）。"""
    import array
    merged = list(ctx) if ctx else []
    merged.extend(samples_list)
    pcm = array.array("h", merged).tobytes()

    async def _run() -> str:
        try:
            t0 = time.time()
            text = await asyncio.to_thread(_transcribe_local, pcm)
            print(f"[asr-vad] segment {len(pcm) // 32}ms -> {time.time() - t0:.2f}s: {text!r}", flush=True)
            return text
        except Exception as e:
            print(f"[asr-vad] segment transcribe failed: {e}", flush=True)
            return ""

    call.seg_tasks.append(asyncio.create_task(_run()))


async def _flush_vad_transcript(call: Call) -> str:
    """speech_end 收尾：未闭合的尾段直接成段 → 等全部段转写（总超时30s）→ 拼接。
    返回空串 = 切段无产出，调用方回落整段识别。"""
    if not ASR_CHUNK or call.vad is None:
        return ""
    # 尾段兜底（9-05 她实测吞尾句案）：只要还有残余音频就识别一次，
    # 生死交给幻听过滤——silero 把轻尾音判成静音时，这里就是最后一道救回的机会。
    # 但要攒够 0.8s 音频才送：阈值降到 0.3 后呼吸会攒进 vad_cur，过短的送识别
    # 就是给 SenseVoice 的气流幻觉（幽灵"okay"）开闸——真尾句 1.5s 左右不受影响
    if call.vad_cur and len(call.vad_cur) >= int(SAMPLE_RATE * 0.8):
        _schedule_segment(call, call.vad_cur, ctx=call.vad_seg_ctx)
    call.vad_cur, call.vad_speech_win, call.vad_silence_win = [], 0, 0
    call.vad_tail, call.vad_seg_ctx = [], []
    tasks = list(call.seg_tasks)
    call.seg_tasks = []
    if not tasks:
        return ""
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30)
    except asyncio.TimeoutError:
        print("[asr-vad] segment gather timeout, using partial results", flush=True)
    texts = []
    for t in tasks:
        if t.done() and not t.cancelled() and not isinstance(t.result(), BaseException):
            r = t.result()
            if r and not _is_hallucination(r):
                texts.append(r.strip())
    transcript = " ".join(texts).strip()
    print(f"[asr-vad] turn joined: {len(texts)}/{len(tasks)} segment(s), {len(transcript)} chars", flush=True)
    return transcript


async def answer_turn(ws, call: Call, http: aiohttp.ClientSession, pcm: bytes,
                      supplied_text: str = "", prev_generation: asyncio.Task | None = None,
                      vad_transcript: str = "") -> None:
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
        # 转写三级优先：打字 > VAD 切段拼接（段级幻听已滤） > 整段识别
        transcript = supplied_text or vad_transcript or await transcribe(http, pcm)
        metrics["asr_done_at"] = time.time()
        if not transcript:
            await send(ws, {"type": "nothing_heard"})
            await _finish_turn(ws, call)
            return
        if not supplied_text and not vad_transcript:  # 打字/切段路径已各自滤过，此处只防整段路径幻听
            if _pcm_rms(pcm) < MIN_SPEECH_RMS or _is_hallucination(transcript):
                print(f"[vad-filter] dropped as hallucination: {transcript!r}", flush=True)
                await send(ws, {"type": "nothing_heard"})
                await _finish_turn(ws, call)
                return
        # 顶替时机：新轮内容确认在手（ASR 非幻听/打字）才掐旧轮——
        # 边缘音频轮、幻听轮从此没有杀人资格（她的实战教训：打字轮两度被陪葬）。
        # prev_generation 由 session 显式传入（创建时刻的旧任务），不会误伤自己。
        if prev_generation is not None and not prev_generation.done():
            prev_generation.cancel()
        # 自然挂断第一步（COVE §16）：这一轮是不是告别？反例保护优先（"别挂"含"挂"字）。
        call.farewell = bool(FAREWELL_RE.search(transcript)) and not FAREWELL_NEG_RE.search(transcript)
        # 先存事实（内存归档缓冲 + 顺序队列落盘），再通知可能已离线的前端（闻序遗漏二）
        call.turns.append(("她", transcript))
        call.writer.submit(turn_seq, turn_id, "user", transcript)
        await send(ws, {"type": "transcript", "call_session_id": call.id, "turn_id": turn_id, "text": transcript})
        call.generation += 1
        generation = call.generation
        metrics["generation_id"] = generation

        streamed = bool(GATEWAY_URL) and STREAM_TTS
        if streamed:
            # V2 流式（COVE §12）：SSE 增量 → 按行切句 → 每段立刻清洗/合成/下发。
            # 首句 4~14 字（K 措辞协议）最先出声，不再等网关全量+整段 TTS。
            # 代价：assistant 轮的存档从"先存后发"变为"流完再存"——SSE 期间被 cancel
            # （打断/挂断）则该轮 reply 不入档，与 M1.5-3"被顶轮生成即止"语义一致。
            seg_parts: list[str] = []
            seg_first_done = False
            tts_fail_streak = 0        # TTS 熔断器（9-05 她 401 风暴案：每段空转 1.5~2s 拖死整轮）
            tts_muted_until = 0.0

            def _is_cjk_line(text: str) -> bool:
                """去标点空白后含中文且非点缀（CJK≥2 字且占比≥1/3）→ 中文行。
                阈值放这么宽是故意的：翻译行常夹英文单词（'你居然 ping 我？' CJK 恰好
                过不了过半线），而漏判的成本是把中文念出来（她实测的 bug），误判的
                成本只是少读一行该静默的字幕。"""
                s = re.sub(r"[\s，。！？、,.!?~～…\-—'\"“”()\[\]：:；;]", "", text)
                if not s:
                    return False
                cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
                return cjk >= 2 and cjk * 3 >= len(s)

            async def _emit_segment(line: str) -> None:
                nonlocal seg_first_done, tts_fail_streak, tts_muted_until
                seg_parts.append(line)
                if generation != call.generation:
                    return  # 已被顶替/打断：字幕音频都不再出（文本照攒，方便日志排查）
                spoken, caption = split_for_tts(line)
                # 双语协议（K 措辞）：英文朗读一行（带引号），中文翻译另起一行/多行（无引号）。
                # 整段模式全文提取引号内容，翻译行天然被排除；流式按行切后翻译行会踩中
                # split_for_tts 的"无引号兜底整行进 TTS"——中文被念出来（9-05 她实测）。
                # 修：无引号且中文为主的行=字幕行，只进字幕不进 TTS；
                # 引号内纯中文同样跳读（K 爱引用她的原词如"不小心"，读出来是 bug
                # 而 ElevenLabs 的英文声线念中文也不在设计内）；混合句保留朗读。
                has_quote = bool(re.search(r'"[^"]+"', line))
                skip_tts = ((not has_quote) and _is_cjk_line(line)) or (bool(spoken) and _is_cjk_line(spoken))
                if caption:
                    await send(ws, {"type": "reply_text", "generation_id": generation,
                                    "turn_id": turn_id, "text": caption})
                if not spoken or skip_tts:
                    return
                if time.time() < tts_muted_until:
                    return  # 熔断中：字幕照常，TTS 跳过不空转
                seg_metrics = None
                if not seg_first_done:   # 延迟指标只看首段（后续段不覆盖 *_at）
                    seg_first_done = True
                    metrics["first_segment_at"] = time.time()
                    seg_metrics = metrics
                try:
                    audio = await synthesize(http, spoken, seg_metrics)
                    tts_fail_streak = 0
                except Exception as e:
                    tts_fail_streak += 1
                    if tts_fail_streak >= 3:
                        tts_muted_until = time.time() + 90   # 连续3败→熔断90s，电话继续（字幕在）
                        tts_fail_streak = 0
                        print(f"[tts-seg] circuit OPEN 90s after 3 fails; last: {e}", flush=True)
                    else:
                        print(f"[tts-seg] failed: {e}", flush=True)  # 单段失败跳过，后续段继续
                    return
                if audio and generation == call.generation:
                    call.set_state(CallState.K_SPEAKING)
                    await send(ws, {"type": "audio", "generation_id": generation,
                                    "data": base64.b64encode(audio).decode("ascii")})
                    await send(ws, {"type": "audio_sentence_end", "generation_id": generation})

            reply = await request_reply(http, {"call_session_id": call.id, "turn_id": turn_id,
                                               "transcript": transcript}, metrics,
                                        on_segment=_emit_segment)
        else:
            reply = await request_reply(http, {"call_session_id": call.id, "turn_id": turn_id,
                                               "transcript": transcript}, metrics)
        if not reply:
            await _finish_turn(ws, call)
            return
        call.turns.append(("他", reply))
        call.writer.submit(turn_seq, turn_id, "assistant", reply)  # 闻序遗漏一：同 turn_id 不同 role，与 user 轮成对
        if not streamed:
            spoken, caption = split_for_tts(reply)   # 引号内朗读段转译方言；字幕用清洗后文本（无协议标记）
            await send(ws, {"type": "reply_text", "generation_id": generation, "turn_id": turn_id,
                            "text": caption})  # 闻序热修：空 caption 不回退原始 reply（防协议标签漏进字幕）
            audio = await synthesize(http, spoken, metrics) if spoken else None
            if audio and generation == call.generation:
                call.set_state(CallState.K_SPEAKING)   # 音频已下发（实际播完由前端自理）
                await send(ws, {"type": "audio", "generation_id": generation, "data": base64.b64encode(audio).decode("ascii")})
                await send(ws, {"type": "audio_sentence_end", "generation_id": generation})
        if streamed:
            metrics["tts_stream_first_ok"] = seg_first_done  # 首段是否出过声（纯观测）
        await send(ws, {"type": "generation_end", "generation_id": generation})
        if call.farewell:
            # 告别回复已下发：告诉前端"道别中"，播完排空后由 graceful_hangup 收线
            metrics["farewell"] = True
            await send(ws, {"type": "hangup_soon", "grace_ms": HANGUP_GRACE_MS})
        await _finish_turn(ws, call)
        await log_metrics(call, turn_seq, turn_id, generation, metrics)  # 纯观测写入，失败不影响通话
    except Exception as error:  # do not serialize credentials or provider bodies
        await _finish_turn(ws, call)  # 收口先行（内含 ws 断保护），再尽量通知错误
        try:
            await send(ws, {"type": "error", "error": str(error)})
        except Exception:
            pass


async def graceful_hangup(ws, call: Call) -> None:
    """自然挂断（COVE §16 / M2）：告别回复已生成下发，这里等三件事再收线——
    ① 前端播放真正排空（playback_idle，即 TTS 队列听不见了的地面信号）
    ② 宽限期内她没再开口（她再开口 = speech_start 分支取消本任务，告别不算数）
    ③ 全程硬截止 HANGUP_DEADLINE_S，超时不等了直接关。
    到点后请前端自行挂断（do_hangup），归档照旧统一走 session 的 finally——一套出口。"""
    print(f"[farewell] graceful hangup armed (grace={HANGUP_GRACE_MS}ms, "
          f"deadline={HANGUP_DEADLINE_S}s)", flush=True)
    try:
        deadline = time.time() + HANGUP_DEADLINE_S
        while not call.playback_idle_at and time.time() < deadline:
            await asyncio.sleep(0.2)
        grace_end = time.time() + HANGUP_GRACE_MS / 1000
        while time.time() < grace_end and time.time() < deadline:
            await asyncio.sleep(0.2)
        call.set_state(CallState.ENDING)
        try:
            await send(ws, {"type": "do_hangup"})
        except Exception:
            pass
        await asyncio.sleep(1.5)   # 给前端留出发送 hangup/本地清理的余裕
        try:
            await ws.close()       # 前端没响应也强制收线；async for 退出 → finally 归档
        except Exception:
            pass
    except asyncio.CancelledError:
        raise  # 她宽限期内又开口了：告别作废，通话继续（session 里已同步复位 call.farewell）


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
                        _vad_feed(call, raw)     # VAD 切段：边说边切边识别（朋友方案）
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
                    # 她又开口了：在途的自然挂断一律作废（说拜拜之后想起还有事说，太正常）
                    if call.hangup_task and not call.hangup_task.done():
                        call.hangup_task.cancel()
                    call.hangup_task = None
                    call.farewell = False
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
                    if kind == "text":
                        # 打字没有 speech_start 前奏（VAD 不触发）：宽限期内她改打字说事，同样算反悔
                        if call.hangup_task and not call.hangup_task.done():
                            call.hangup_task.cancel()
                        call.hangup_task = None
                        call.farewell = False
                        pcm = b""
                    else:
                        pcm = call.end_turn()
                        # VAD 误触发防御：过短/过低音量的"轮"不创建任务、不惊动任何人
                        if len(pcm) < SAMPLE_RATE * 2 * 0.3 or _pcm_rms(pcm) < MIN_SPEECH_RMS:
                            print(f"[vad-filter] junk turn dropped "
                                  f"({len(pcm) // 3200}0ms, rms={_pcm_rms(pcm):.0f})", flush=True)
                            call.set_state(CallState.LISTENING)
                            await send(ws, {"type": "state", "mode": "listening"})
                            continue
                    # M1.5-3 核心：生成任务后台化，接收循环不再被 ASR/网关/TTS 阻塞。
                    # 顶替不在这里做——此刻还不知道新轮有没有真话，answer_turn 在内容确认后才掐旧轮。
                    vad_transcript = await _flush_vad_transcript(call) if kind == "speech_end" else ""
                    prev_task = call.pending_generation
                    pending_generation = asyncio.create_task(
                        answer_turn(ws, call, http, pcm, str(event.get("text", "")) if kind == "text" else "",
                                    prev_generation=prev_task if isinstance(prev_task, asyncio.Task) else None,
                                    vad_transcript=vad_transcript))
                    call.pending_generation = pending_generation  # 同步赋值，先于新任务首次调度

                    def _arm_farewell(task: asyncio.Task) -> None:
                        """告别轮正常落幕后才武装收线（answer_turn 吞异常，cancelled 除外）。
                        她中途再开口会在 speech_start 复位 farewell，回调到时自然哑火。"""
                        if call.farewell and not task.cancelled() and (call.hangup_task is None or call.hangup_task.done()):
                            call.playback_idle_at = 0.0  # 旧轮的排空信号作废——必须等告别回复自己播完
                            call.hangup_task = asyncio.create_task(graceful_hangup(ws, call))
                    pending_generation.add_done_callback(_arm_farewell)
                elif kind == "playback_idle":
                    call.playback_idle_at = time.time()   # 前端播放排空：自然挂断等的就是这个地面信号
                elif kind == "interrupt":
                    if call.hangup_task and not call.hangup_task.done():
                        call.hangup_task.cancel()      # 打断告别轮 = 告别作废
                    call.hangup_task = None
                    call.farewell = False
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
            if call.hangup_task and not call.hangup_task.done():
                call.hangup_task.cancel()    # 自然挂断任务拿着 ws，同理必须收割在 ClientSession 之前
                try:
                    await asyncio.wait({call.hangup_task}, timeout=1)
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
