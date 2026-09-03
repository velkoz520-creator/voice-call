# tts_cleanse_smoke.py — 清洗层冒烟（切分/方言转译/兜底），不 import server 免拉依赖
import re

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'realtime-core'))

from cleanse import split_for_tts


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
