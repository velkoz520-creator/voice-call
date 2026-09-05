# PaiVoice · Jester 魔改版（电话项目）

> 给 AI 伴侣（住在网关里的人格）打电话的完整链路：浏览器拨号 → 本服务当耳朵和嘴 → **网关语音快车道**里的大脑接电话。
>
> **上游血统**：基于 [tianyupaipai-cmd/pai-voice](https://github.com/tianyupaipai-cmd/pai-voice)（AGPL-3.0，本仓库继承同协议）。感谢原作者的通话底座。
>
> 维护者：Jester · 2026-09-01 立项 · 本版 2026-09-06（M2/V2 全量后）

---

## 一、架构：三层各司其职

```
她（手机浏览器 / packages/web-client/index.html）
   │ getUserMedia 麦克风 + WebSocket（PCM16 16k 实时流）
   ▼
本服务（packages/realtime-core/server.py，Zeabur 容器）
   │ 耳朵：本地 SenseVoice int8 ONNX（sherpa-onnx，容器内推理）
   │   └ silero-vad 边说边切段（段闭合即后台识别——说完秒出，延迟不随说话时长涨）
   │ 大脑：POST 网关语音快车道（OpenAI SSE），流式增量 → 按行切句
   │ 嘴：ElevenLabs 逐段合成（首句不等全量）+ TTS 熔断器
   ▼
网关语音快车道（独立仓库 tiantian-wg / voice_lane.py，不在本仓库）
   │ 人格注入 + 通话缓存 + OB 记忆检索（首句一次）+ 语音槽渠道（五槽面板可切）
   ▼
渠道（LLM）→ 回复原路返回 → TTS → 声音 + 双语字幕
```

**大脑和历史的单一事实源在网关侧**：本服务每轮只发 `{call_session_id, transcript}`，
不保存对话。前端换、耳嘴换，网关不动——整个系统的设计基石。

## 二、演进史速览（每一代都是实测喂出来的）

| 代 | 内容 |
|---|---|
| M0/M1 | 链路打通 + 对讲（橘瓣特殊版通话 + 网关快车道） |
| M1.5 | 延迟分阶段指标 / 逐轮落盘（TurnWriter）/ 生成任务后台化 + 状态机 / 预滚缓冲防吞句首 / 幻听过滤 + 打字条 |
| M2 | 自适应停句（900/1350ms）/ 两阶段打断（240ms duck → 520ms interrupt，误触 160ms restore）/ 自然挂断（告别词 → 播完 → 宽限 3.5s → 自动收线，30s 硬截止） |
| V2 | **首句切分流式 TTS**（SSE 增量按行切句逐段合成，首轮 8~12s → 2~3s） |
| 9-05 | **ASR 本地化**（SenseVoice 进容器，跨太平洋航线停飞，5.5s → 秒级）+ **silero-vad 切段** + 六轮实测修复（吞字/幽灵词/音量/中文误读） |

## 三、代码地图

| 文件 | 职责 |
|---|---|
| `packages/realtime-core/server.py` | 全部服务端：WS 会话 / 本地 ASR + VAD 切段 / 网关流式 / 流式 TTS / 自然挂断 / 逐轮落盘 / 指标 |
| `packages/realtime-core/cleanse.py` | 清洗层：`split_for_tts`（引号切分+方言转译）、`LineSegmenter`（流式按行切句）、`ThinkFilter`（流式剥 `<think>` 推理块） |
| `packages/web-client/voice-call.js` | 通话核心类：采集/VAD（自适应停句+两阶段打断+duck）/播放队列（分句 flush）/预滚/打字条/麦克风健康观察/播放增益+压缩器 |
| `packages/web-client/index.html` | 拨号页：双语字幕渲染 / 状态灯 / 打字条 / 静音横幅 / 自动挂断接线 |
| `Dockerfile` | Zeabur 构建；**构建期从 HF 拉模型进镜像**（SenseVoice int8 230MB） |

`voice-call.js` 早已不是上游原样（M1.5 起深度魔改），升级上游时只能参考不能覆盖。

## 四、WS 协议（前后端契约，维护必读）

**前端 → 服务端**
| 消息 | 说明 |
|---|---|
| `start` `{video, sample_rate}` | 接通（可带 token） |
| 二进制帧 | PCM16 16k 实时音频 |
| `speech_start` `{barge, preroll}` | 开口；preroll=开口前 ~0.9s 预滚（防吞句首） |
| `speech_end` | 说完（触发 VAD 段收割 → 网关） |
| `text` `{text}` | 打字条（不打断播放，排队） |
| `interrupt` | 前端确认打断（服务端 cancel 在途任务+作废音频） |
| `playback_idle` | 播放队列排空（自然挂断等的信号） |
| `hangup` | 挂断 |

**服务端 → 前端**
`state`（含 call_session_id）/ `transcript`（她的转写）/ `reply_text`（清洗后字幕）/ `audio`+`audio_sentence_end`（分句音频）/ `generation_end` / `interrupted` / `nothing_heard`（没听清→字幕提示）/ `hangup_soon`（道别中）/ `do_hangup`（请收线）/ `error`

## 五、耳朵：本地 ASR + VAD 切段（9-05 上线）

- **SenseVoiceSmall int8 ONNX**（sherpa-onnx）跑在容器内，`ASR_PROVIDER=local`。
  本机实测 ~100x 实时；Zeabur 2vCPU 秒级。异常自动回落 SiliconFlow（同一只耳朵的云备份）。
- **silero-vad 切段**（`ASR_CHUNK=1`）：静音 ≥0.7s 闭合一段 → 后台识别存着 →
  说完只剩尾段。**仅 provider=local 时生效**（段识别写死本地模型）。
- 模型文件三重保障：镜像构建期下载（主）→ 运行时自愈下载（`_ensure_model_file`，urllib 标准库）→ 云端回落。
- **调参史（别乱动，都是她实测换来的）**：threshold 0.5→0.30（轻尾音误判静音=吞尾句）；
  min_silence 0.7s；起音垫 0.4s 上下文（silero 复位后头几窗判定滞后=吞句首）；
  flush 兜底 ≥0.8s 才送（呼吸段送识别=幽灵 "okay"）。

## 六、嘴：流式 TTS + 三重防线

- **流式**（`STREAM_TTS=1`）：网关 SSE 增量 → `ThinkFilter` 剥推理 → `LineSegmenter` 按行切 →
  每段独立清洗/合成/下发。K 的输出协议一行=一句英文+中文翻译，行即切分单位。
- **清洗防线**：引号内=朗读、引号外=字幕；**引号内纯中文跳读**（他爱引用她的原词，
  "不小心"被念案）；语气中间协议转译各家方言（换厂商只改 `cleanse.py` 的 `_TTS_DIALECT`）。
- **熔断器**：连续 3 段失败停 90s（字幕照常电话不断）；报错带响应体（401 品种直接可读——
  2026-09-05 曾因额度耗尽 401 风暴 19 分钟）。
- 播放端：`playGain` 2.2x + DynamicsCompressor 兜峰值（她要求原生更大声）；duck 打断压至 25%。

## 七、部署（Zeabur）+ zbpack 大坑

**⚠️ zbpack-v2 会永久缓存"预处理后的 Dockerfile"**——git push 触发的构建永远用该服务
第一次构建时的指令结构，只更新 COPY 的文件。**改 Dockerfile 结构（加 RUN/ENV 层）必须走
规格化部署**（MCP `deploy-from-specification`，内联 Dockerfile 直传）；改 .py/.js/.html 走
正常 push 即可。规格化部署一次后缓存重置，普通 push 恢复正常（2026-09-05 实测）。

环境变量（真 Key 只放这里，绝不入库）：

| 变量 | 说明 |
|---|---|
| `PAIVOICE_TOKEN` | 拨号令牌（前端填同一个） |
| `PAIVOICE_ASR_PROVIDER` | `local`（主）｜`siliconflow`（云端/回落）｜`groq`｜`mock` |
| `PAIVOICE_ASR_API_KEY` | 云端 ASR key（local 的回落保险） |
| `PAIVOICE_ASR_CHUNK` | `1`=silero 切段（仅 local 时生效） |
| `PAIVOICE_TTS_PROVIDER` / `PAIVOICE_TTS_API_KEY` | `elevenlabs` |
| `PAIVOICE_ELEVEN_VOICE_ID` / `PAIVOICE_ELEVEN_MODEL` / `PAIVOICE_ELEVEN_STABILITY` | 音色 / `eleven_v3` / 0.0~1.0 |
| `PAIVOICE_STREAM_TTS` | `1`=流式切分（出问题置 0 秒回整段） |
| `PAIVOICE_GATEWAY_URL` / `PAIVOICE_GATEWAY_TOKEN` | 网关快车道 + `API_SECRET` |
| `PAIVOICE_ARCHIVE_URL` | 网关归档端点 |
| `PAIVOICE_SB_URL` / `PAIVOICE_SB_KEY` | Supabase（逐轮落盘+指标） |
| `PAIVOICE_HANGUP_GRACE_MS` / `PAIVOICE_HANGUP_DEADLINE_S` | 自然挂断宽限/硬截止 |

**额度账本**：ElevenLabs 按档位月字符额度，三方共用（电话+橘瓣+TG）。一通 2.5h 电话可吃掉
Starter 全月 30k——长电话前查余额（2026-09-05 事故）。

## 八、网关侧约定（联调契约）

- 分流：UA 带 `pai-voice`，body 带 `call_session_id`；首句全量上下文（含记忆检索）后走缓存；`[接通了]`=接通标记。
- 归档：挂断 POST 全文 → `voice_calls` 表 → 挂断即摘要（K 口吻写 OB）+ `voice_call_turns` 逐轮表（本服务实时写，断线零丢失）+ `voice_call_metrics` 分段延迟指标。
- 语音槽：网关五槽面板（聊天/识图/压缩/语音/日记）可一键切语音渠道——渠道断供她自己救，不用等 Jester 改库。

## 九、测试与升级

- 本仓库测试脚本一律临时写、跑完删（历史惯例；冒烟样例见 git log 各 fix 提交信息）
- 本地快速验证：`PAIVOICE_LOCAL_ASR_DIR` 指到本地模型目录 + 起 server + WS 推 wav（样例见 9-05 提交记录）
- 测试模型/音频在 `_model_test/`（未入库）——换机器按 README §五 的 URL 重下
- 升级上游 pai-voice：server.py 与 voice-call.js 均为深度魔改件，只参考不覆盖

## 十、许可

AGPL-3.0（继承上游）。自部署自用。
