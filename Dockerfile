FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY packages/realtime-core/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/realtime-core/server.py ./server.py
COPY packages/realtime-core/cleanse.py ./cleanse.py
COPY packages/web-client/index.html ./index.html
COPY packages/web-client/voice-call.js ./voice-call.js
# 容器本地 ASR（9-05 天天拍板）：SenseVoiceSmall int8 ONNX，构建期从 HF 拉取
# （Ashburn→HF 快线；模型随镜像走，运行时零下载依赖）。体积 +230MB 换 ASR 免跨洋
ENV PAIVOICE_LOCAL_ASR_DIR=/app/asr-model
RUN mkdir -p /app/asr-model     && curl -fsSL --retry 3 -o /app/asr-model/model.int8.onnx        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx"     && curl -fsSL --retry 3 -o /app/asr-model/tokens.txt        "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt"
# build marker: 强制后续层缓存失效（89b225f+ = 幻听过滤+打字条版）
RUN echo "build-ref: 89b225f-hallucination-filter-textbar" > /app/.build_ref
ENV PAIVOICE_HOST=0.0.0.0
EXPOSE 8780
CMD ["python", "server.py"]
