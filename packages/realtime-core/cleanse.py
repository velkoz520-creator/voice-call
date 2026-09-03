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
