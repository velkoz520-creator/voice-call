# tts_cleanse_smoke.py — 清洗层冒烟（切分/方言转译/兜底），不 import server 免拉依赖
import re

_TTS_DIALECT = {
    "elevenlabs": {"[laughs]": "[laughs]", "(pause)": "...", "(sighs)": "[sighs]",
                   "[whispers]": "[whispers]", "(laughs)": "[laughs]", "(sighs)": "[sighs]"},
    "minimax": {"(pause)": "<#0.6#>", "[laughs]": "哈哈", "(laughs)": "哈哈",
                "[sighs]": "唉", "(sighs)": "唉", "[whispers]": ""},
}


def split_for_tts(text, provider):
    dialect = _TTS_DIALECT.get(provider, {})
    quoted = re.findall(r'"([^"]*)"', text)
    spoken = " ".join(q.strip() for q in quoted if q.strip()) if quoted else text
    # 顺序铁律：markdown 清理必须在方言转换之前（否则会吃掉 MiniMax 的 <#0.6#> 原生标记）
    spoken = re.sub(r"[*_`#>]+", "", spoken)
    for s, d in dialect.items():
        spoken = spoken.replace(s, d)
    if provider != "elevenlabs":
        spoken = re.sub(r"\[[^\]\n]{1,20}\]", "", spoken)
    spoken = spoken.strip()
    caption = re.sub(r"\[[^\]\n]{1,20}\]", "", text)
    caption = re.sub(r"\([^)\n]{1,16}\)", "", caption)
    caption = re.sub(r"[*_`#>]+", "", caption)
    return spoken, caption


reply = '"Hey, it\'s me. [laughs] Took you long enough (pause) kidding."'
reply += "\n是我。等你好久了（笑），开玩笑的。"

sp, cap = split_for_tts(reply, "elevenlabs")
assert "[laughs]" in sp, "elevenlabs 方言直通失败: " + sp
assert "..." in sp, "(pause) 转译失败: " + sp
assert "是我。" in cap, "字幕丢了翻译行: " + cap
assert "（笑）" in cap, "全角括号是中文表达，字幕应保留: " + cap

# 半角协议标记即使出现在中文行（措辞异常）也该被剥掉，防被念出来
sp_bad, cap_bad = split_for_tts('"Okay." (laughs) 好啦(laughs)。', "elevenlabs")
assert "(laughs)" not in cap_bad, "半角标记残留字幕: " + cap_bad

sp2, cap2 = split_for_tts(reply, "minimax")
assert "[laughs]" not in sp2 and "哈哈" in sp2, "minimax 笑声转译失败: " + sp2
assert "(pause)" not in sp2 and "<#0.6#>" in sp2, "minimax 停顿转译失败: " + sp2

noq = "没有引号的异常输出"
assert split_for_tts(noq, "elevenlabs")[0] == noq, "无引号兜底失败"

empty = '"  "\n（只有空格）'
sp3, _ = split_for_tts(empty, "mock")
assert sp3 == "" or "只有空格" in sp3, "空引号兜底异常: " + repr(sp3)

print("清洗层冒烟 5/5 ✅")
