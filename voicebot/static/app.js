/**
 * OmniVoice Bot — Frontend Application
 *
 * Uses Whisper ASR on the server for speech-to-text (instead of Web Speech API).
 * Flow: Record mic audio → Send to /api/voice-chat → Receive bot audio response.
 */

// ============================================================
// State
// ============================================================

const state = {
    isRecording: false,
    isProcessing: false,
    mediaRecorder: null,
    audioChunks: [],
    currentAudio: null,
    messageCount: 0,
    stream: null,
};

// ============================================================
// DOM Elements
// ============================================================

const chatArea = document.getElementById('chatArea');
const textInput = document.getElementById('textInput');
const micBtn = document.getElementById('micBtn');
const sendBtn = document.getElementById('sendBtn');
const statusDot = document.querySelector('.status-dot');
const statusText = document.querySelector('.status-text');
const welcomeMessage = document.getElementById('welcomeMessage');
const voiceStyle = document.getElementById('voiceStyle');

// ============================================================
// Microphone Recording (MediaRecorder API)
// ============================================================

async function initMicrophone() {
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
        console.log('Microphone access granted');
        return true;
    } catch (err) {
        console.error('Microphone access denied:', err);
        setStatus('Mic access denied', 'error');
        return false;
    }
}

function startRecording() {
    if (!state.stream) return;

    state.audioChunks = [];

    // Use webm format (widely supported)
    const options = { mimeType: 'audio/webm;codecs=opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        // Fallback
        delete options.mimeType;
    }

    state.mediaRecorder = new MediaRecorder(state.stream, options);

    state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.audioChunks.push(event.data);
        }
    };

    state.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
        state.audioChunks = [];

        // Only process if we have meaningful audio (> 1KB)
        if (audioBlob.size > 1000) {
            await sendVoiceMessage(audioBlob);
        } else {
            setStatus('Recording too short', 'ready');
            setTimeout(() => setStatus('Ready', 'ready'), 2000);
        }
    };

    state.mediaRecorder.start(100); // Collect data every 100ms
    state.isRecording = true;
    updateUI();
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        state.mediaRecorder.stop();
    }
    state.isRecording = false;
    updateUI();
}

// ============================================================
// UI Functions
// ============================================================

function setStatus(text, type = 'ready') {
    statusText.textContent = text;
    statusDot.className = 'status-dot';
    if (type === 'listening') statusDot.classList.add('listening');
    if (type === 'processing') statusDot.classList.add('processing');
}

function updateUI() {
    if (state.isRecording) {
        micBtn.classList.add('active');
        micBtn.querySelector('.mic-icon').classList.add('hidden');
        micBtn.querySelector('.stop-icon').classList.remove('hidden');
        setStatus('🎤 Recording... (click to stop)', 'listening');
    } else if (state.isProcessing) {
        micBtn.classList.remove('active');
        micBtn.querySelector('.mic-icon').classList.remove('hidden');
        micBtn.querySelector('.stop-icon').classList.add('hidden');
        setStatus('🧠 Typhoon ASR + TTS ...', 'processing');
    } else {
        micBtn.classList.remove('active');
        micBtn.querySelector('.mic-icon').classList.remove('hidden');
        micBtn.querySelector('.stop-icon').classList.add('hidden');
        setStatus('Ready', 'ready');
    }
}

function hideWelcome() {
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
}

function addMessage(text, type, audioBlob = null, meta = {}) {
    hideWelcome();
    state.messageCount++;

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;
    msgDiv.id = `msg-${state.messageCount}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    content.appendChild(bubble);

    // Audio player for bot messages
    if (audioBlob && type === 'bot') {
        const audioDiv = createAudioPlayer(audioBlob);
        content.appendChild(audioDiv);
    }

    // Meta info
    if (meta.latency || meta.language || meta.asrMs || meta.ttsMs) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        if (meta.asrMs && meta.asrMs !== '0') {
            const asrSpan = document.createElement('span');
            asrSpan.textContent = `🎤 ASR ${Math.round(parseFloat(meta.asrMs))}ms`;
            metaDiv.appendChild(asrSpan);
        }
        if (meta.ttsMs && meta.ttsMs !== '0') {
            const ttsSpan = document.createElement('span');
            ttsSpan.textContent = `🔊 TTS ${Math.round(parseFloat(meta.ttsMs))}ms`;
            metaDiv.appendChild(ttsSpan);
        }
        if (meta.latency) {
            const latencySpan = document.createElement('span');
            latencySpan.textContent = `⚡ Total ${Math.round(parseFloat(meta.latency))}ms`;
            metaDiv.appendChild(latencySpan);
        }
        if (meta.language) {
            const langSpan = document.createElement('span');
            langSpan.textContent = `🌐 ${meta.language === 'th' ? 'Thai' : 'English'}`;
            metaDiv.appendChild(langSpan);
        }
        content.appendChild(metaDiv);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    chatArea.appendChild(msgDiv);

    // Auto scroll
    chatArea.scrollTop = chatArea.scrollHeight;

    return msgDiv;
}

function createAudioPlayer(audioBlob) {
    const audioDiv = document.createElement('div');
    audioDiv.className = 'message-audio';

    const playBtn = document.createElement('button');
    playBtn.className = 'play-btn';
    playBtn.innerHTML = '▶';

    const waveDiv = document.createElement('div');
    waveDiv.className = 'audio-wave';

    // Create wave bars
    for (let i = 0; i < 16; i++) {
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.style.height = `${4 + Math.random() * 12}px`;
        waveDiv.appendChild(bar);
    }

    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    playBtn.onclick = () => {
        if (audio.paused) {
            // Stop any currently playing audio
            if (state.currentAudio && state.currentAudio !== audio) {
                state.currentAudio.pause();
                state.currentAudio.currentTime = 0;
                document.querySelectorAll('.play-btn.playing').forEach(btn => {
                    btn.classList.remove('playing');
                    btn.innerHTML = '▶';
                });
                document.querySelectorAll('.audio-wave.playing').forEach(w => {
                    w.classList.remove('playing');
                });
            }

            audio.play();
            state.currentAudio = audio;
            playBtn.classList.add('playing');
            playBtn.innerHTML = '⏸';
            waveDiv.classList.add('playing');
        } else {
            audio.pause();
            playBtn.classList.remove('playing');
            playBtn.innerHTML = '▶';
            waveDiv.classList.remove('playing');
        }
    };

    audio.onended = () => {
        playBtn.classList.remove('playing');
        playBtn.innerHTML = '▶';
        waveDiv.classList.remove('playing');
        state.currentAudio = null;
    };

    audioDiv.appendChild(playBtn);
    audioDiv.appendChild(waveDiv);

    return audioDiv;
}

function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message bot';
    msgDiv.id = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    bubble.appendChild(typing);

    content.appendChild(bubble);
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    return msgDiv;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

// ============================================================
// Voice Chat (Whisper ASR Pipeline)
// ============================================================

async function sendVoiceMessage(audioBlob) {
    if (state.isProcessing) return;

    // Show user message (we don't know the text yet — ASR will tell us)
    const userMsg = addMessage('🎤 (transcribing...)', 'user');

    state.isProcessing = true;
    updateUI();
    addTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.webm');
        formData.append('voice_style', voiceStyle.value || '');

        const response = await fetch('/api/voice-chat', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Get metadata from headers
        const userText = decodeHeaderValue(response.headers.get('X-User-Text') || '');
        const botText = decodeHeaderValue(response.headers.get('X-Bot-Text') || 'Response received');
        const language = response.headers.get('X-Language') || 'en';
        const latency = response.headers.get('X-Latency-Ms') || '0';
        const asrMs = response.headers.get('X-ASR-Ms') || '0';
        const ttsMs = response.headers.get('X-TTS-Ms') || '0';

        // Update the user message with actual transcribed text
        const userBubble = userMsg.querySelector('.message-bubble');
        if (userBubble) {
            userBubble.textContent = userText || '(no speech detected)';
        }

        // Get audio blob
        const responseAudioBlob = await response.blob();

        // Remove typing and add bot message
        removeTypingIndicator();

        if (userText && userText !== '(empty)') {
            addMessage(botText, 'bot', responseAudioBlob, {
                latency,
                language,
                asrMs,
                ttsMs,
            });

            // Auto-play
            autoPlayAudio(responseAudioBlob);
        } else {
            addMessage('ไม่ได้ยินเสียง ลองพูดใหม่อีกครั้งครับ', 'bot');
        }

    } catch (err) {
        removeTypingIndicator();
        // Update user message
        const userBubble = userMsg.querySelector('.message-bubble');
        if (userBubble) userBubble.textContent = '🎤 (failed)';
        addMessage(`Error: ${err.message}. Make sure the server is running on port 8002.`, 'bot');
        console.error('Voice chat error:', err);
    } finally {
        state.isProcessing = false;
        updateUI();
    }
}

// ============================================================
// Text Chat (fallback — no ASR needed)
// ============================================================

async function sendTextMessage() {
    const text = textInput.value.trim();
    if (!text || state.isProcessing) return;

    addMessage(text, 'user');
    textInput.value = '';

    state.isProcessing = true;
    updateUI();
    addTypingIndicator();

    try {
        const response = await fetch('/api/chat-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                voice_style: voiceStyle.value || null,
            }),
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const botText = decodeHeaderValue(response.headers.get('X-Bot-Text') || 'Response received');
        const language = response.headers.get('X-Language') || 'en';
        const latency = response.headers.get('X-Latency-Ms') || '0';
        const ttsMs = response.headers.get('X-TTS-Ms') || '0';

        const audioBlob = await response.blob();
        removeTypingIndicator();

        addMessage(botText, 'bot', audioBlob, {
            latency,
            language,
            ttsMs,
        });

        autoPlayAudio(audioBlob);

    } catch (err) {
        removeTypingIndicator();
        addMessage(`Error: ${err.message}`, 'bot');
    } finally {
        state.isProcessing = false;
        updateUI();
    }
}

// ============================================================
// Helpers
// ============================================================

function decodeHeaderValue(val) {
    try {
        // Handle latin-1 encoded UTF-8
        const bytes = new Uint8Array([...val].map(c => c.charCodeAt(0)));
        return new TextDecoder('utf-8').decode(bytes);
    } catch {
        return val;
    }
}

function autoPlayAudio(audioBlob) {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    state.currentAudio = audio;

    audio.onended = () => {
        state.currentAudio = null;
        document.querySelectorAll('.play-btn.playing').forEach(btn => {
            btn.classList.remove('playing');
            btn.innerHTML = '▶';
        });
        document.querySelectorAll('.audio-wave.playing').forEach(w => {
            w.classList.remove('playing');
        });
    };

    audio.play().catch(() => {
        console.log('Auto-play blocked by browser policy');
    });
}

function sendQuickMessage(text) {
    textInput.value = text;
    sendTextMessage();
}

// ============================================================
// Event Handlers
// ============================================================

async function toggleMic() {
    if (state.isProcessing) return;

    if (state.isRecording) {
        stopRecording();
    } else {
        if (!state.stream) {
            const granted = await initMicrophone();
            if (!granted) {
                alert('ไม่สามารถเข้าถึงไมโครโฟนได้\nกรุณาอนุญาตการใช้ไมค์ใน Browser Settings');
                return;
            }
        }
        startRecording();
    }
}

// Enter key to send text
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendTextMessage();
    }
});

// Send button
sendBtn.addEventListener('click', () => sendTextMessage());

// ============================================================
// Initialize
// ============================================================

console.log('OmniVoice Bot initialized (Whisper ASR mode)');
setStatus('Ready — Click 🎤 to speak', 'ready');
