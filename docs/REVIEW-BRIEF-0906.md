# 闻序审查任务书 · voice-call 9-05 六代迭代全量复审

> 交给人：闻序（GPT）
> 仓库：velkoz520-creator/voice-call（私有），main 分支
> 审查范围：`git log 15bcab4` 往前到 `92b1f4e` 的全部提交（9-05 一天的六代迭代），
> 外加 010010d（耦合修复+README）。核心文件：`packages/realtime-core/server.py`、
> `packages/realtime-core/cleanse.py`、`packages/web-client/voice-call.js`、`Dockerfile`。
> Jester 自检已跑：py_compile / node --check / VAD 五场景 / 熔断四场景 / ThinkFilter
> 八场景 / provider 耦合回归——全绿。**要挑的是自检看不见的。**

## 背景（30 秒版）

AI 伴侣电话：浏览器 PCM16 流 → 本服务（耳朵=容器内 SenseVoice ASR + silero-vad
切段；大脑=网关 LLM 流式；嘴=ElevenLabs 流式 TTS 分句下发）。9-05 一天迭代六代：
本地 ASR → silero 切段 → 起音垫料 → 幽灵词过滤 → 音量增益 → 耦合修复。线上在跑，
她实测驱动的修复为主。

## 审查重点（按优先级）

1. **并发与时序**（最要紧）：
   - `seg_tasks`（VAD 段转写任务）的生命周期：`begin_turn` 清引用、`_flush_vad_transcript`
     收割——与 `pending_generation` 顶替、`hangup_task` 自然挂断、连接断开（finally）
     的交叉时序有没有洞？段任务会不会活过 ClientSession？
   - `_vad_feed` 在 WS 接收循环里同步跑（含 silero 推理）——长段会不会阻塞收流？
   - `_local_decode_lock` 串行 decode：段转写任务并发排队时，speech_end 的
     `asyncio.wait_for(..., 30)` 超时后未完成任务去哪了？
2. **状态机边界**：`ThinkFilter`/`LineSegmenter` 的跨 chunk 断标签、`<think>` 未闭合、
   空增量；`VadModel.reset()` 后的窗口状态残留；`vad_tail`/`vad_seg_ctx` 在
   begin_turn/interrupt/断连时的清理完整性。
3. **回落路径完整性**：local→siliconflow 回落、VAD→整段回落、流式→整段（STREAM_TTS=0）、
   熔断开启时的行为。回落触发时 metrics/落盘/归档是否一致。
4. **协议正确性**：`answer_turn` 的 farewell/顶替/generation 守卫在 VAD 路径下
   （`vad_transcript` 非空时）的分支组合；`skip_tts`（中文跳读）与字幕下发的互斥。
5. **Dockerfile/env**：模型构建期下载失败的可观测性；env 缺失时的行为。

## 红线（不要碰/不要建议改）

- `threshold=0.30`、`min_silence=0.7`、起音垫 0.4s、flush 0.8s 门槛——全是她实测
  事故换来的数字，每降低一档都有对应事故，**不要建议"调回去"**。
- 回落与总开关设计（STREAM_TTS/ASR_CHUNK/provider 回落）是 B6 三保险，别建议删除。
- 任何修改走 PR/patch 建议，**不要直接 push**（push=部署线上，必须天天发话）。

## 输出格式

问题列表，每条：`[P0/P1/P2] 文件:行号 — 问题 — 建议修法`。P0=可能线上炸，
P1=边界洞或数据不一致，P2=风格/健壮性建议。没有问题的区域明确说"查过没问题"，
别硬凑。
