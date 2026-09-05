/* 实时语音/视频通话 · 浏览器端（无依赖 ES module）
 *
 * 一条媒体流三种用途：分析流（VAD）、识别流（推给后端转写）、——原声不归档（隐私：音频不落盘）。
 * 即使顾川在说话，麦克风也继续听：她一开口就 interrupt，他立刻停。
 *
 * 用法：
 *   const call = new VoiceCall({ url: 'wss://your-pai-voice.example/voice/ws', video: false, on: {...} });
 *   await call.start();   // 必须在用户点击里调用（iOS 需要手势解锁 AudioContext）
 *   call.hangup();
 */

const WORKLET_SRC = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.len = 0; this.acc = 0; this.ratio = sampleRate / 16000; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // 线性抽样到 16k（AEC/NS 已在 getUserMedia 做过）
    const out = [];
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) { this.acc -= this.ratio; out.push(ch[i]); }
    }
    let sum = 0;
    for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
    const rms = Math.sqrt(sum / ch.length);
    const i16 = new Int16Array(out.length);
    for (let i = 0; i < out.length; i++) { const v = Math.max(-1, Math.min(1, out[i])); i16[i] = v < 0 ? v * 32768 : v * 32767; }
    this.buf.push(i16); this.len += i16.length;
    if (this.len >= 1024) {                       // ~64ms @16k
      const all = new Int16Array(this.len); let o = 0;
      for (const b of this.buf) { all.set(b, o); o += b.length; }
      this.buf = []; this.len = 0;
      this.port.postMessage({ pcm: all.buffer, rms }, [all.buffer]);
    } else {
      this.port.postMessage({ rms });
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
`;

export class VoiceCall {
  constructor({ url, video = false, on = {}, vad = {} } = {}) {
    this.url = url;
    this.wantVideo = video;
    this.on = on;
    this.vad = Object.assign({
      startHold: 3,        // 连续几块（~20ms/块）超阈值算开口
      // 自适应停句（COVE §7 / M2）：固定阈值两难——短了思考停顿被抢断，长了短句收口拖沓。
      // 短句说完 900ms 收口，长发言（有声累计>1.8s）给 1350ms；单句 60s 硬上限从起音算。
      endSilenceShortMs: 900,
      endSilenceLongMs: 1350,
      longSpeechMs: 1800,  // 本轮有声累计超过这个 = 长发言，用长停句阈值
      maxTurnMs: 60000,    // 单句硬上限（从起音算，不看"进入聆听"时刻）
      // 两阶段打断（COVE §14 / M2）：扬声器回声/咳嗽/桌面碰撞不再误杀他的话——
      // 240ms 连续人声先 duck（压低他的播放音量，留住证据），520ms 才 interrupt 真打断；
      // 误触消失 160ms 内 restore 回原音量。预卷缓冲（M1.5-4）保证打断不吃她的句首。
      duckMs: 240,
      interruptMs: 520,
      restoreMs: 160,
      duckGain: 0.25,      // duck 时他的音量压到基准的 25%
      playGain: 2.2,       // 播放基准增益（9-05 她反馈1.5仍小；配压缩器防爆音）
      floor: 0.006,        // 最低噪声底
      gain: 3.2,           // 进入阈值 = max(floor, 噪声底 * gain)
      exitRatio: 0.62,     // 结束阈值 = 进入阈值 * 这个（双阈值，防止悬在临界卡住）
      speakingGain: 2.8,   // 顾川说话时再提高门槛，降低扬声器误触
      watchdogMs: 1600,    // 这么久没有一块声音超过进入阈值 → 强制收尾（AGC 把环境音抬高也不会卡）
    }, vad);

    this.ws = null; this.ctx = null; this.stream = null; this.node = null;
    this.mode = 'idle';      // idle | listening | thinking | speaking
    this.speaking = false;   // 她在说
    this.playing = false;    // 他在说（扬声器）
    // 预滚缓冲（M1.5-4）：VAD 确认开口要 ~200ms，确认前的音频服务端没存——
    // 常备 ~900ms 环形缓冲，speech_start 时随消息补发，ASR 永远听到完整的她
    this._preroll = []; this._prerollLen = 0;
    this.noise = 0.01; this._hot = 0; this._silenceSince = 0; this._hotSince = 0; this._lastHotAt = 0; this._speechMin = 1;
    this._lastTickAt = 0; this._voicedMs = 0; this._speechStartedAt = 0;
    this._ducked = false; this._duckSilentSince = 0;   // 两阶段 barge-in 证据
    this._lastVoiceAt = 0; this._micHinted = false;    // 麦克风健康观察（误触静音检测）
    this.generationId = null;
    this._queue = []; this._pending = new Map(); this._sources = []; this._playhead = 0;
    this._playGen = 0;   // 播放取消序号：每次 stop/hangup 递增，在途解码完成对不上即弃（闻序 9-06 P1 #11）
    this._pendingDecodes = 0;   // 在途解码计数：完成信号=无在途解码+无播放源（闻序二轮 #2）
    this.video = null; this._frameTimer = null; this.canvas = null;
    this.callSessionId = null; this.stats = { turns: 0, firstAudioMs: null };
    this._turnSentAt = 0;
  }

  emit(ev, ...a) { try { this.on[ev] && this.on[ev](...a); } catch (e) { console.error(e); } }

  // ------------------------------------------------------------ 开始
  async start() {
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await this.ctx.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      video: this.wantVideo ? { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } : false,
    });
    const blob = new Blob([WORKLET_SRC], { type: 'application/javascript' });
    const workletUrl = URL.createObjectURL(blob);
    await this.ctx.audioWorklet.addModule(workletUrl);
    this.outGain = this.ctx.createGain();
    this.outGain.gain.value = this.vad.playGain;   // 播放基准增益（9-05 她反馈1.5仍小，提至2.2）
    // 压缩器兜峰值：推大响度的同时防削波爆音
    this.comp = this.ctx.createDynamicsCompressor();
    this.comp.threshold.value = -24; this.comp.knee.value = 20;
    this.comp.ratio.value = 6; this.comp.attack.value = 0.005; this.comp.release.value = 0.15;
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.outGain.connect(this.comp);
    this.comp.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this._ampBuf = new Uint8Array(this.analyser.fftSize);
    const ampLoop = () => {
      if (!this.ctx || this.ctx.state === 'closed') return;
      if (this.playing && this.analyser) {
        this.analyser.getByteTimeDomainData(this._ampBuf);
        let sum = 0;
        for (let i = 0; i < this._ampBuf.length; i++) { const v = (this._ampBuf[i] - 128) / 128; sum += v * v; }
        this.emit('himLevel', Math.sqrt(sum / this._ampBuf.length));
      }
      requestAnimationFrame(ampLoop);
    };
    requestAnimationFrame(ampLoop);
    const src = this.ctx.createMediaStreamSource(new MediaStream([this.stream.getAudioTracks()[0]]));
    this.node = new AudioWorkletNode(this.ctx, 'pcm-capture');
    this.node.port.onmessage = (e) => this._onCapture(e.data);
    src.connect(this.node);
    // worklet 不接到 destination，避免自己听见自己

    if (this.wantVideo && this.stream.getVideoTracks().length) {
      this.video = document.createElement('video');
      this.video.muted = true; this.video.playsInline = true; this.video.autoplay = true;
      this.video.srcObject = new MediaStream([this.stream.getVideoTracks()[0]]);
      await this.video.play().catch(() => {});
      this.emit('video', this.video);
    }

    await this._connect();
    this._send({ type: 'start', video: this.wantVideo, sample_rate: 16000 });
    this._setMode('listening');
    if (this.video) this._frameTimer = setInterval(() => this._pushFrame(), 5000);
  }

  _connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { this.ws = ws; resolve(); };
      ws.onerror = () => reject(new Error('连不上通话服务'));
      ws.onclose = (e) => { this.ws = null; if (this.mode !== 'idle') { this._setMode('idle'); this.emit('closed', e); } };
      ws.onmessage = (e) => this._onMessage(e.data);
    });
  }

  _send(obj) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }

  _setMode(m) { if (this.mode !== m) { this.mode = m; this.emit('state', m); } }

  // ------------------------------------------------------------ 听（Listen 轨 + VAD）
  _onCapture({ pcm, rms }) {
    if (pcm) {
      // worklet 传来的是 ArrayBuffer（transfer 后），统一转 Int16Array——
      // 闻序 9-06 P1：此前直接 push ArrayBuffer，.length=undefined → 预滚恒为空串
      const samples = (pcm instanceof Int16Array) ? pcm : new Int16Array(pcm);
      this._pushPreroll(samples);                                       // 预滚缓冲常转（M1.5-4）
      if (this.ws && this.ws.readyState === 1) this.ws.send(pcm);       // 二进制 PCM16 照常流式上传
    }
    if (rms === undefined) return;
    this.emit('level', rms);

    const now = performance.now();
    const dt = this._lastTickAt ? Math.min(now - this._lastTickAt, 200) : 0;  // 帧间隔（封顶防挂起后暴涨）
    this._lastTickAt = now;
    // 噪声底：她没说话时快跟；说话中也慢慢向观测到的最小值靠（AGC 抬底噪时不至于卡死）
    if (!this.speaking) this.noise = this.noise * 0.97 + rms * 0.03;
    else { this._speechMin = Math.min(this._speechMin, rms); this.noise = Math.min(this.noise * 1.0015, Math.max(this.noise, this._speechMin)); }
    let enter = Math.max(this.vad.floor, this.noise * this.vad.gain);
    if (this.playing) enter *= this.vad.speakingGain;
    const exit = enter * this.vad.exitRatio;

    if (rms > enter) {
      this._lastHotAt = now;
      this._lastVoiceAt = now; this._micHinted = false;
    }

    // 麦克风健康观察：聆听中连续 90 秒收不到任何超过门槛的声音——大概率误触了
    // 静音键（9-05 她 98 分钟打字案的头号嫌疑）。只提示一次，开口即复位。
    if (!this.speaking && this._lastVoiceAt && now - this._lastVoiceAt >= 90000 && !this._micHinted) {
      this._micHinted = true;
      this.emit('micSilent');
    }

    if (!this.speaking) {
      if (rms > enter) {
        this._hot += 1;
        if (!this._hotSince) this._hotSince = now;
        if (this._hot >= this.vad.startHold) {
          if (this.playing && this.mode === 'speaking') {
            // 两阶段 barge-in（COVE §14）：240ms 先 duck 留证据，520ms 才真打断
            const voiced = now - this._hotSince;
            if (!this._ducked && voiced >= this.vad.duckMs) { this._ducked = true; this._duckTo(this.vad.playGain * this.vad.duckGain); }
            if (voiced >= this.vad.interruptMs) { this._duckTo(this.vad.playGain); this._speechStart(); }
          } else {
            this._speechStart();
          }
        }
      } else if (this._ducked) {
        // 误触消失（回声/咳嗽）：160ms 内恢复他的原音量，证据清零重来
        if (!this._duckSilentSince) this._duckSilentSince = now;
        else if (now - this._duckSilentSince >= this.vad.restoreMs) {
          this._ducked = false; this._duckSilentSince = 0; this._duckTo(this.vad.playGain);
          this._hot = 0; this._hotSince = 0;
        }
      } else { this._hot = 0; this._hotSince = 0; }
      return;
    }
    // 说话中：自适应停句（COVE §7）——短句 900ms 快收口，长发言 1350ms 给思考停顿留白；
    // 或者看门狗——太久没有真正的语音峰值也收尾
    if (rms >= exit) {
      this._voicedMs += dt;
      this._silenceSince = 0;
    } else {
      if (!this._silenceSince) this._silenceSince = now;
      const need = this._voicedMs < this.vad.longSpeechMs ? this.vad.endSilenceShortMs : this.vad.endSilenceLongMs;
      if (now - this._silenceSince >= need) { this._speechEnd(); return; }
    }
    if (this._speechStartedAt && now - this._speechStartedAt >= this.vad.maxTurnMs) { this._speechEnd(); return; }
    if (this._lastHotAt && now - this._lastHotAt >= this.vad.watchdogMs) this._speechEnd();
  }

  /** duck/restore：只动他这一路的 outGain，她的麦克风不受影响。 */
  _duckTo(v) {
    if (!this.outGain || !this.ctx) return;
    const t = this.ctx.currentTime;
    try {
      this.outGain.gain.cancelScheduledValues(t);
      this.outGain.gain.setTargetAtTime(v, t, 0.04);
    } catch { /* ignore */ }
  }

  // ------------------------------------------------------------ 预滚（M1.5-4）
  _pushPreroll(samples) {
    this._preroll.push(samples); this._prerollLen += samples.length;
    const cap = 15 * 1024;                     // 15 块 × 1024 样本 @16k ≈ 960ms
    while (this._prerollLen > cap && this._preroll.length > 1) {
      this._prerollLen -= this._preroll[0].length;
      this._preroll.shift();
    }
  }

  _prerollB64() {
    if (!this._preroll.length) return null;
    const total = this._preroll.reduce((n, b) => n + b.length, 0);
    const all = new Int16Array(total); let o = 0;
    for (const b of this._preroll) { all.set(b, o); o += b.length; }
    const bytes = new Uint8Array(all.buffer);
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  _speechStart() {
    this.speaking = true; this._speechMin = 1; this._lastHotAt = performance.now(); this._silenceSince = 0;
    this._speechStartedAt = performance.now(); this._voicedMs = 0;   // 自适应停句：本轮有声累计从零起算
    this._ducked = false; this._duckSilentSince = 0;
    this._lastVoiceAt = performance.now(); this._micHinted = false;
    this.emit('speech', true);
    // 只在听得到正式回复时打断；思考中、短提示音或误触都让他继续准备。
    const barge = this.playing && this.mode === 'speaking';
    if (barge) {
      this._stopPlayback();
      this._send({ type: 'interrupt', turn_id: 'new' });
    }
    this._send({ type: 'speech_start', barge, preroll: this._prerollB64() });
  }

  _speechEnd() {
    this.speaking = false; this._silenceSince = 0;
    this.emit('speech', false);
    this._turnSentAt = performance.now();
    this._send({ type: 'speech_end' });
    this._setMode('thinking');
  }

  /** 不用麦克风也能试：直接发文字。
   *  他正在说话时不打断——播放中的音频不受影响，新回复的音频天然排在他念完之后
   *  （排队，简单版：真实电话里也是等他说完才接话）。 */
  sendText(text) {
    this._turnSentAt = performance.now();
    this._send({ type: 'text', text });
    this._setMode('thinking');
  }

  // ------------------------------------------------------------ 收
  async _onMessage(data) {
    let msg; try { msg = JSON.parse(data); } catch { return; }
    switch (msg.type) {
      case 'state':
        if (msg.call_session_id) this.callSessionId = msg.call_session_id;
        if (msg.tts) this.emit('info', msg);
        if (msg.mode) this._setMode(msg.mode === 'speaking' && !this.playing ? 'thinking' : msg.mode);
        break;
      case 'transcript': this.stats.turns += 1; this.emit('transcript', msg); break;
      case 'prosody': this.emit('prosody', msg); break;
      case 'reply_text':
        if (msg.generation_id !== this.generationId) { this.generationId = msg.generation_id; }
        this.emit('reply', msg); break;
      case 'audio': this._onAudio(msg); break;
      case 'audio_sentence_end': this._flushSentence(msg.generation_id); break;
      case 'generation_end': this._flushSentence(msg.generation_id); this._pending.delete(msg.generation_id); break;
      case 'interrupted': this._stopPlayback(); this.emit('interrupted', msg); break;
      case 'nothing_heard': this.emit('nothingHeard', msg); break;
      case 'hangup_soon': this.emit('hangupSoon', msg); break;    // 告别轮：他正在说再见
      case 'do_hangup': this.emit('autoHangup', msg); break;      // 宽限期过完：服务端请我们挂断
      case 'observation': this.emit('observation', msg); break;
      case 'error': this.emit('error', msg.error); break;
      default: break;
    }
  }

  _onAudio(msg) {
    if (msg.kind === 'ack') {
      if (!this.playing && this._queue.length === 0) this._decodeAndQueue(msg.data, msg.generation_id, true);
      return;
    }
    if (!this.generationId || msg.generation_id >= this.generationId) this.generationId = msg.generation_id;
    if (msg.generation_id !== this.generationId) return;          // 旧轮的丢掉
    const key = msg.generation_id;
    if (!this._pending.has(key)) this._pending.set(key, []);
    this._pending.get(key).push(msg.data);
  }

  _flushSentence(genId) {
    const parts = this._pending.get(genId);
    if (!parts || !parts.length) return;
    this._pending.set(genId, []);
    const bins = parts.map((b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
    const total = bins.reduce((n, b) => n + b.length, 0);
    const all = new Uint8Array(total); let o = 0;
    for (const b of bins) { all.set(b, o); o += b.length; }
    this._decodeAndQueue(all.buffer, genId, false);
  }

  async _decodeAndQueue(dataOrB64, genId, isAck) {
    const myGen = this._playGen;   // 闻序 P1 #11：解码是异步的，打断后旧解码完成不得重新入队
    this._pendingDecodes++;
    let buf = dataOrB64;
    if (typeof dataOrB64 === 'string') buf = Uint8Array.from(atob(dataOrB64), (c) => c.charCodeAt(0)).buffer;
    let audio;
    try { audio = await this.ctx.decodeAudioData(buf.slice(0)); }
    catch (e) { this._pendingDecodes--; console.warn('解码失败', e); return; }
    this._pendingDecodes--;
    if (myGen !== this._playGen) return;   // 解码期间被打断/挂断：丢弃
    if (!isAck && genId !== this.generationId) return;
    this._queue.push({ audio, genId, isAck });
    this._drain();
  }

  _drain() {
    const now = this.ctx.currentTime;
    if (this._playhead < now) this._playhead = now + 0.02;
    while (this._queue.length) {
      const { audio, genId } = this._queue.shift();
      const src = this.ctx.createBufferSource();
      src.buffer = audio; src.connect(this.outGain || this.ctx.destination);
      src.start(this._playhead);
      this._playhead += audio.duration;
      this._sources.push(src);
      src.onended = () => {
        this._sources = this._sources.filter((s) => s !== src);
        if (!this._sources.length) this._checkPlaybackIdle();
      };
      if (!this.playing) {
        this.playing = true; this.emit('playing', true);
        if (this._turnSentAt && this.stats.firstAudioMs === null) this.stats.firstAudioMs = Math.round(performance.now() - this._turnSentAt);
        if (this._turnSentAt) { this.emit('latency', Math.round(performance.now() - this._turnSentAt)); this._turnSentAt = 0; }
      }
      if (genId === this.generationId) this._setMode('speaking');
    }
  }

  /** 闻序二轮 #2：完成信号 = 播放源空 + 无在途解码 + 队列空——三者到齐才上报，
   *  每次状态变化都重判（告别音频段间空隙不会误发，播完最后一刻必然触发）。 */
  _checkPlaybackIdle() {
    if (this._sources.length || this._pendingDecodes > 0 || this._queue.length > 0) return;
    if (!this.playing) return;
    this.playing = false; this.emit('playing', false);
    this._send({ type: 'playback_idle', generation_id: this.generationId });
    if (this.mode === 'speaking') this._setMode('listening');
  }

  _stopPlayback() {
    this._playGen++;   // 使在途 decodeAudioData 全部作废
    for (const s of this._sources) { try { s.onended = null; s.stop(); } catch { /* ignore */ } }
    this._sources = []; this._queue = []; this._pending.clear();
    this._playhead = 0;
    this._duckTo(this.vad.playGain); this._ducked = false; this._duckSilentSince = 0;   // 打断后他的音量必须回到原位
    if (this.playing) { this.playing = false; this.emit('playing', false); }
  }

  // ------------------------------------------------------------ 看（Phase 2）
  _pushFrame() {
    if (!this.video || !this.ws || this.ws.readyState !== 1 || this.video.videoWidth === 0) return;
    if (!this.canvas) this.canvas = document.createElement('canvas');
    const w = 480, h = Math.round(this.video.videoHeight * (w / this.video.videoWidth));
    this.canvas.width = w; this.canvas.height = h;
    this.canvas.getContext('2d').drawImage(this.video, 0, 0, w, h);
    const dataUrl = this.canvas.toDataURL('image/jpeg', 0.6);
    this._send({ type: 'frame', data: dataUrl.split(',')[1], ts: Date.now() });
  }

  // ------------------------------------------------------------ 挂断
  hangup() {
    this._send({ type: 'hangup' });
    this._stopPlayback();
    if (this._frameTimer) clearInterval(this._frameTimer);
    if (this.node) { try { this.node.disconnect(); } catch { /* ignore */ } }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close().catch(() => {});
    if (this.ws) { try { this.ws.close(); } catch { /* ignore */ } }
    this.ws = null; this.stream = null; this.node = null; this.video = null;
    this._setMode('idle');
  }
}

export default VoiceCall;
