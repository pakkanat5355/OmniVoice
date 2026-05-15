# Voicebot + Asterisk Setup Guide

## Architecture Flow

```
SIP Phone (ext 1002, 1003...)
    │
    │  dials 9000
    ▼
Asterisk PBX  (172.18.72.117)
    │  [internal] context → extension 9000
    │  app.Answer()
    │  app.AudioSocket("00000000-...,127.0.0.1:8003")
    ▼
asterisk_proxy.py  (127.0.0.1:8003)
    │  AudioSocket TCP → WebSocket bridge
    │  receives 8kHz 16-bit PCM from Asterisk
    │  forwards to voicebot via WebSocket
    ▼
ngrok tunnel  (wss://paraxial-houston-tawnily.ngrok-free.dev)
    ▼
voicebot/server.py  (Windows machine, port 8002)
    │  /asterisk_ws endpoint
    │  1. VAD detects speech end (silence > 0.8s)
    │  2. Groq API → Whisper ASR → Thai transcription
    │  3. Rule-based chatbot → response text
    │  4. gTTS → Thai speech audio (24kHz float32)
    │  5. Resample 24kHz → 8kHz PCM → send back
    ▼
asterisk_proxy.py  (sends audio back to Asterisk)
    ▼
SIP Phone  (hears bot response)
```

---

## Files Changed / Created

### On Asterisk Server (172.18.72.117)

| File | What changed |
|------|-------------|
| `/etc/asterisk/extensions.lua` | Added `[internal]` context with extension 9000 (AudioSocket to port 8003). Also fixed pre-existing syntax bug (missing `};` closing the `["local"]` table). |
| `/home/trbsysadmin/applications/voicebot/asterisk_proxy.py` | Already existed — unchanged. Started as background process. |
| `/home/trbsysadmin/applications/voicebot/proxy.log` | Created at startup — live log of proxy connections. |

### On Windows Machine (local)

| File | Purpose |
|------|---------|
| `voicebot/server.py` | Main voicebot FastAPI server — run this to start the bot |
| `voicebot/asterisk_proxy.py` | Local copy of proxy (reference only — the server copy is used) |
| `voicebot/static/index.html` | Web UI for browser-based testing |
| `voicebot/start_ngrok.py` | Starts ngrok tunnel to expose port 8002 |

---

## Key Config Values

### Asterisk Dialplan (`/etc/asterisk/extensions.lua`)

```lua
extensions["internal"] = {
    ["9000"] = function()
        app.Answer()
        -- UUID must be 36-char format (AudioSocket requirement)
        app.AudioSocket("00000000-0000-0000-0000-000000000009,127.0.0.1:8003")
        app.Hangup()
    end;
}
```

- Context: `internal` (matches `context=internal` in pjsip.conf for all SIP endpoints)
- Extension: `9000`
- AudioSocket port: `8003` (proxy listens here)

### Proxy (on Asterisk server)

```bash
python3 -u asterisk_proxy.py \
  --port 8003 \
  --ws-url wss://paraxial-houston-tawnily.ngrok-free.dev/asterisk_ws
```

- Listens on: `127.0.0.1:8003` (AudioSocket TCP from Asterisk)
- Connects to: ngrok WebSocket URL → voicebot `/asterisk_ws`

### Voicebot Server (`voicebot/server.py`)

| Setting | Value |
|---------|-------|
| Port | `8002` |
| ASR | Groq API (Whisper large-v3, Thai) |
| TTS | gTTS (Google TTS, Thai/English) |
| WebSocket endpoint | `/asterisk_ws` |
| Audio format in | 8kHz 16-bit PCM (from Asterisk via proxy) |
| Audio format out | 8kHz 16-bit PCM (back to Asterisk) |
| VAD silence threshold | 40 chunks × 20ms = **0.8 seconds** |
| VAD energy threshold | `400` (tune down if bot misses quiet speech) |

---

## How to Start Everything

### Step 1 — Start voicebot server (Windows machine)

```bash
uv run python voicebot/server.py
```

### Step 2 — Start ngrok tunnel (Windows machine)

```bash
ngrok http 8002
# or
uv run python voicebot/start_ngrok.py
```

Copy the `wss://xxxx.ngrok-free.app` URL.

### Step 3 — Start proxy on Asterisk server

```bash
ssh trbsysadmin@172.18.72.117

nohup python3 -u ~/applications/voicebot/asterisk_proxy.py \
  --port 8003 \
  --ws-url wss://YOUR-NGROK-URL/asterisk_ws \
  > ~/applications/voicebot/proxy.log 2>&1 &
```

> **Note:** ngrok free tier changes URL on every restart.
> Update `--ws-url` each time ngrok restarts.

### Step 4 — Call extension 9000

Dial `9000` from any SIP phone registered to Asterisk.

---

## Monitoring

```bash
# Watch proxy connections live (on Asterisk server)
tail -f ~/applications/voicebot/proxy.log

# Check proxy is running
ps aux | grep asterisk_proxy

# Check AudioSocket port is open
ss -tlnp | grep 8003

# Verify dialplan via AMI
python3 -c "
import socket, time
s = socket.socket(); s.connect(('127.0.0.1', 5038)); s.settimeout(2); s.recv(1024)
s.sendall(b'Action: Login\r\nUsername: monitor\r\nSecret: password\r\n\r\n'); time.sleep(0.3); s.recv(4096)
s.sendall(b'Action: Command\r\nCommand: dialplan show 9000@internal\r\n\r\n'); time.sleep(1)
print(s.recv(4096).decode())
"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Call drops immediately | Proxy not running | Start proxy, check port 8003 |
| Proxy log empty | Output buffering | Use `python3 -u` flag |
| Proxy connects but no audio | Voicebot server not running | Start `voicebot/server.py` |
| Bot doesn't respond | Groq API key issue or VAD not triggering | Check server.py logs, lower `_VAD_ENERGY_THRESHOLD` |
| Dialplan not found | pbx_lua not reloaded | Run `module reload pbx_lua.so` via AMI |
| ngrok URL changed | Free tier limitation | Restart proxy with new URL |

---

## Why These Choices Were Made

| Decision | Reason |
|----------|--------|
| Edit `extensions.lua` (not `extensions.conf`) | `pbx_config.so` (which loads extensions.conf) was **Not Running** — the live dialplan is handled entirely by `pbx_lua.so` |
| Use hardcoded UUID in AudioSocket | `UNIQUEID` like `1778855675.0` is not 36 chars — AudioSocket requires strict UUID format |
| AudioSocket proxy instead of direct WebSocket | Asterisk's AudioSocket speaks TCP binary protocol; a bridge is needed to convert to WebSocket |
| gTTS instead of OmniVoice for bot responses | OmniVoice requires heavy GPU/model load — gTTS is fast and lightweight for real-time calls |
| Groq API for ASR | Faster than running Whisper locally; optimized for Thai with the right prompt |
