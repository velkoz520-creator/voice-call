# PaiVoice · Jester 魔改版（电话项目）

> 给 AI 伴侣（住在网关里的人格）打电话的完整链路：浏览器 PWA 拨号 → 本服务当耳朵和嘴 → **网关语音快车道**里的大脑接电话。
>
> **上游血统**：基于 [tianyupaipai-cmd/pai-voice](https://github.com/tianyupaipai-cmd/pai-voice)（AGPL-3.0，本仓库继承同协议）。感谢原作者的通话底座——VAD 参数和温和打断是实战调过的真金。
>
> 维护者：Jester · 2026-09-01

---

## 一、架构：三层各司其职

```
她（手机浏览器 / PWA 拨号页 packages/web-client/index.html）
   │ getUserMedia 麦克风 + WebSocket（PCM16 16k）
   ▼
本服务（packages/realtime-core/server.py）
   │ 耳朵 ASR：SiliconFlow SenseVoiceSmall（中文主赛道）
   │ 大脑：POST 网关语音快车道（见下），消费 OpenAI SSE 聚合
   │ 嘴 TTS：ElevenLabs（主赛道）｜MiniMax（桩位 M1 后接）
   ▼
网关语音快车道（独立仓库 tiantian-wg / voice_lane.py，不在本仓库）
   │ 人格上下文注入（lean）+ 通话缓存 + 意图分流 + OB 记忆检索（首句一次）
   ▼
渠道（LLM）→ 回复原路返回 → TTS → 声音 + 双语字幕
```

**大脑和历史的单一事实源在网关侧**：本服务每轮只发 `{call_session_id, transcript}`，
不保存对话（仅挂断归档时暂存全文）。前端换、耳嘴换，网关不动——这是整个系统的设计基石。

## 二、相对上游的魔改清单（维护必读）

| 文件 | 改动 |
|---|---|
| `packages/realtime-core/server.py` | **重写**：ASR 加 siliconflow；Adapter 加 **gateway 模式**（OpenAI+SSE，UA=`pai-voice/0.1` 供网关分流）；新增**清洗层 `split_for_tts()`**；新增**挂断归档** `archive_call()`；移除无用 numpy |
| `packages/web-client/index.html` | **新增**：通话页（拨号/状态灯/金色圆盘音量/**双语字幕分轨渲染**/延迟显示/静音挂断） |
| `Dockerfile` | **新增**：Zeabur 部署用 |
| `.env.example` | **重写**：全量 env 清单 |
| `requirements.txt` | 移除 numpy（全包零引用） |
| `tts_cleanse_smoke.py` | **新增**：清洗层冒烟测试 |

`packages/web-client/voice-call.js` **保持上游原样**（VAD/温和打断/播放队列/延迟秒表是实战参数，别动）。

## 三、清洗层：语气中间协议（重要机制）

措辞（存在网关侧）里**只使用中间协议**：`[laughs]` `[sighs]` `[whispers]` / `(pause)` `(laughs)` `(sighs)`。
本服务的 `split_for_tts()` 在合成前做两件事：

1. **切分**：引号内 = 朗读段；引号外 = 字幕段（剥半角协议标记，**全角中文括号保留**——那是中文表达不是协议）。
2. **转译**：按 `_TTS_DIALECT` 映射表把中间协议转成当前厂商方言（ElevenLabs 直通；MiniMax `(pause)`→`<#0.6#>`）。

**换 TTS 厂商的步骤**：`_TTS_DIALECT` 加该家映射 → `synthesize()` 加该家分支 → 跑 `tts_cleanse_smoke.py`。措辞零改动。
**顺序铁律**：markdown 清理必须在方言转换**之前**（否则会吃掉 MiniMax 的 `<#0.6#>` 原生标记——冒烟实测抓过）。

## 四、部署（Zeabur）

1. 新建 Zeabur 项目 → 部署本仓库（自动识别 Dockerfile）
2. 环境变量（真 Key 只放这里，绝不入库）：

| 变量 | 说明 |
|---|---|
| `PAIVOICE_TOKEN` | 浏览器拨号令牌（自定随机串，通话页里填同一个） |
| `PAIVOICE_ASR_PROVIDER` | `siliconflow` |
| `PAIVOICE_ASR_API_KEY` | 硅基流动 key |
| `PAIVOICE_SILICONFLOW_ASR_MODEL` | `FunAudioLLM/SenseVoiceSmall` |
| `PAIVOICE_TTS_PROVIDER` | `elevenlabs` |
| `PAIVOICE_TTS_API_KEY` | ElevenLabs key |
| `PAIVOICE_ELEVEN_VOICE_ID` | 音色 ID |
| `PAIVOICE_GATEWAY_URL` | `https://<网关域名>/v1/chat/completions` |
| `PAIVOICE_GATEWAY_TOKEN` | 网关 `API_SECRET` |
| `PAIVOICE_ARCHIVE_URL` | `https://<网关域名>/v1/voice/archive` |

3. 网关侧（另一仓库）需同步：`VOICE_LANE_ENABLED=1` + `VOICE_CALL_MODE_PROMPT`（电话模式措辞终版）
4. 浏览器打开 `index.html`（静态托管或本地），填 `wss://<本服务域名>/voice/ws` + TOKEN → 接通

## 五、网关侧约定（联调契约）

- 分流：本服务所有请求 UA 带 `pai-voice`，body 带 `call_session_id`——网关据此进快车道
- 首句全量：通话第一句时网关做全量上下文准备（含记忆检索），之后复用缓存；`[接通了]` 是接通标记（她刚接起，给第一声）
- 归档：挂断时本服务 POST `transcript`（全文）到网关 `/v1/voice/archive` → 存 `voice_calls` 表（`pending_summary=true`），摘要由 K 自己写（C3 拍板）

## 六、测试与升级

- 清洗层冒烟：`python tts_cleanse_smoke.py`（5/5 过为绿）
- 升级上游 pai-voice 时对齐 §二 清单：server.py 是重写件（整体 diff），voice-call.js 取上游更新需回归字幕协议与打断行为
- 协议变更史：措辞终版 v1.0（2026-09-01，aigroup id=95~103 决策链）

## 七、许可

AGPL-3.0（继承上游）。自部署自用。
