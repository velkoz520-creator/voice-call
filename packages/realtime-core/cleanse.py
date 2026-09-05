# cleanse.py — TTS 清洗层（生产实现，测试直接 import 本模块，杜绝复制漂移）
# 职责：按 K 的输出协议切分整段回复 → (tts_text, caption_text)。
#   - 引号内 = 朗读段：中间协议语气标记转译成当前厂商方言后进 TTS
#   - 引号外 = 字幕段：只做剥标记清理，进字幕不进 TTS
#   - 无引号时兜底整段进 TTS（措辞异常时宁可多读不错过）
# 中间协议：[laughs] [sighs] [whispers] / (pause) (laughs) (sighs)——措辞只学这一套，
# 各家方言映射在 _TTS_DIALECT，换厂商措辞零改动。

import os
import re

TTS_PROVIDER = os.getenv("PAIVOICE_TTS_PROVIDER", "mock")

# 语气中间协议 → 各家 TTS 方言映射（比武时逐家实测校准）
_TTS_DIALECT = {
    "elevenlabs": {
        # ElevenLabs v3 原生支持 [laughs] [sighs] [whispers] 等方括号 audio tags → 直通；
        # (pause) 无原生停顿标签 → 省略号近似（v3 不支持 SSML break，官方推荐省略号）
        "[laughs]": "[laughs]", "[sighs]": "[sighs]", "[whispers]": "[whispers]",
        "(pause)": "...", "(sighs)": "[sighs]", "(laughs)": "[laughs]",
    },
    "minimax": {
        # MiniMax t2a_v2 原生停顿标记 <#秒#>（0.01~3s）；笑声类暂转文本，比武时校准
        "(pause)": "<#0.6#>", "[laughs]": "哈哈", "(laughs)": "哈哈",
        "[sighs]": "唉", "(sighs)": "唉", "[whispers]": "",
    },
    "mock": {},
}


def split_for_tts(text: str, provider: str | None = None) -> tuple[str, str]:
    """按协议切分整段回复 → (tts_text, caption_text)。
    顺序铁律：markdown 清理必须在方言转换之前（否则会吃掉 MiniMax 的 <#0.6#> 原生标记）。"""
    provider = provider or TTS_PROVIDER
    dialect = _TTS_DIALECT.get(provider, {})

    quoted = re.findall(r'"([^"]*)"', text)
    spoken = " ".join(q.strip() for q in quoted if q.strip()) if quoted else text

    spoken = re.sub(r"[*_`#>]+", "", spoken)
    for src, dst in dialect.items():
        spoken = spoken.replace(src, dst)
    if provider != "elevenlabs":  # 非直通厂商：剥掉残余的方括号标签，防被念出来
        spoken = re.sub(r"\[[^\]\n]{1,20}\]", "", spoken)
    spoken = spoken.strip()

    caption = re.sub(r"\[[^\]\n]{1,20}\]", "", text)
    caption = re.sub(r"\([^)\n]{1,16}\)", "", caption)
    caption = re.sub(r"[*_`#>]+", "", caption)
    return spoken, caption


class ThinkFilter:
    """流式剥离 <think>...</think> 推理块（9-05 她实测案：thinking 渠道把内心独白
    混进 content 流，被切句朗读）。字符级状态机，标签跨 chunk 断开也安全：
    尾部保留可能是半截标签的字符，直到能判定是标签还是普通文本。
    用在 LineSegmenter 之前——字幕、TTS、归档三处一起干净。"""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.in_think = False
        self._buf = ""

    def _partial_tag_len(self, s: str) -> int:
        """s 尾部若挂着某个标签的前缀（如 '<thi'），返回该前缀长度，否则 0。"""
        for tag in (self.OPEN, self.CLOSE):
            for k in range(min(len(s), len(tag) - 1), 0, -1):
                if s.endswith(tag[:k]):
                    return k
        return 0

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        pos = 0
        while pos < len(self._buf):
            if self.in_think:
                j = self._buf.find(self.CLOSE, pos)
                if j >= 0:
                    pos = j + len(self.CLOSE)          # 吞掉推理块，跳到闭合标签之后
                    self.in_think = False
                else:
                    keep = self._partial_tag_len(self._buf[pos:])
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    return "".join(out)                 # 还在推理块里：全部吞掉
            else:
                j = self._buf.find(self.OPEN, pos)
                if j >= 0:
                    out.append(self._buf[pos:j])
                    pos = j + len(self.OPEN)
                    self.in_think = True
                else:
                    keep = self._partial_tag_len(self._buf[pos:])
                    safe_end = len(self._buf) - keep
                    if safe_end > pos:
                        out.append(self._buf[pos:safe_end])
                    self._buf = self._buf[safe_end:]
                    return "".join(out)
        self._buf = ""
        return "".join(out)

    def flush(self) -> str:
        """流结束：缓冲里剩的是非标签正文则放行；未闭合的 <think> 残余属于泄漏
        推理，吞掉不留尾巴。"""
        rest, self._buf = self._buf, ""
        if self.in_think:
            return ""
        return rest


class LineSegmenter:
    """V2 流式 TTS 的增量切句器（COVE §12）：网关 SSE 增量喂进来，切出"可说单元"立刻回调。
    K 的输出协议天然按行分句（一行 = "English line." 中文翻译），所以行就是切分单位；
    无换行的长段按 max_chars 强切兜底（措辞异常时仍能流水出声，不至于憋到全量）。
    段间顺序由消费方保证（串行下发）。"""

    def __init__(self, max_chars: int = 240):
        self.buf = ""
        self.max_chars = max_chars

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        segs: list[str] = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                segs.append(line.strip())
        if len(self.buf) >= self.max_chars:      # 措辞异常兜底：无换行长段强切
            if self.buf.strip():
                segs.append(self.buf.strip())
            self.buf = ""
        return segs

    def flush(self) -> str | None:
        out = self.buf.strip()
        self.buf = ""
        return out or None
