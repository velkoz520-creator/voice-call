FROM python:3.12-slim
WORKDIR /app
COPY packages/realtime-core/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/realtime-core/server.py ./server.py
COPY packages/web-client/index.html ./index.html
COPY packages/web-client/voice-call.js ./voice-call.js
# build marker: 强制后续层缓存失效（5f3f046+ = voice_lane 快车道版）
RUN echo "build-ref: 5f3f046-voice-lane" > /app/.build_ref
ENV PAIVOICE_HOST=0.0.0.0
EXPOSE 8780
CMD ["python", "server.py"]
