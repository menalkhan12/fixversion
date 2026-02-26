let room = null;
let sessionId = null;
let callStartMs = null;
let durationTimer = null;
let qaCount = 0;
let livekitUrl = null;

const statusEl = document.getElementById('status');
const callBtn = document.getElementById('callBtn');
const durationEl = document.getElementById('duration');
const qaCountEl = document.getElementById('qaCount');

const chatInput = document.getElementById('chatInput');
const askBtn = document.getElementById('askBtn');
const chatOut = document.getElementById('chatOut');

function setStatus(s) {
  statusEl.textContent = s;
}

function fmt(ms) {
  const sec = Math.floor(ms / 1000);
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function startTimer() {
  stopTimer();
  durationTimer = setInterval(() => {
    if (!callStartMs) return;
    durationEl.textContent = fmt(Date.now() - callStartMs);
  }, 500);
}

function stopTimer() {
  if (durationTimer) {
    clearInterval(durationTimer);
    durationTimer = null;
  }
}

async function getToken() {
  const res = await fetch('/livekit/token');
  if (!res.ok) throw new Error('token');
  return res.json();
}

async function loadConfig() {
  if (livekitUrl !== null) return;
  const res = await fetch('/config');
  if (!res.ok) {
    livekitUrl = '';
    return;
  }
  const data = await res.json();
  livekitUrl = (data.livekit_url || '').trim();
}

async function startCall() {
  await loadConfig();
  const { token } = await getToken();
  const url = (livekitUrl || '').trim();
  if (!url) {
    throw new Error('LIVEKIT_URL missing');
  }

  room = new LivekitClient.Room({
    adaptiveStream: true,
    dynacast: true,
  });

  room.on(LivekitClient.RoomEvent.Connected, () => {
    setStatus('IST Agent is Listening...');
  });

  room.on(LivekitClient.RoomEvent.Disconnected, () => {
    setStatus('Ready');
  });

  room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === 'audio') {
      const el = track.attach();
      el.autoplay = true;
      document.body.appendChild(el);
    }
  });

  setStatus('Connecting...');
  await room.connect(url, token);

  setStatus('Publishing microphone...');
  await room.localParticipant.setMicrophoneEnabled(true);

  callStartMs = Date.now();
  startTimer();
}

async function endCall() {
  if (room) {
    await room.disconnect();
    room = null;
  }
  callStartMs = null;
  stopTimer();
}

callBtn.addEventListener('click', async () => {
  try {
    if (!room) {
      callBtn.textContent = 'End Call';
      callBtn.classList.add('pulse');
      await startCall();
    } else {
      callBtn.textContent = 'Start Call';
      callBtn.classList.remove('pulse');
      await endCall();
    }
  } catch (e) {
    callBtn.textContent = 'Start Call';
    callBtn.classList.remove('pulse');
    setStatus('Ready');
    chatOut.textContent = 'Setup required: set LIVEKIT_URL on window or serve it via config.';
  }
});

askBtn.addEventListener('click', async () => {
  const text = (chatInput.value || '').trim();
  if (!text) return;

  setStatus('IST Agent is Responding...');

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, text }),
  });

  const data = await res.json();
  sessionId = data.session_id;
  chatOut.textContent = data.answer;
  qaCount += 1;
  qaCountEl.textContent = String(qaCount);

  if (data.escalate) {
    setStatus('Lead Capture Required');
  } else {
    setStatus('IST Agent is Listening...');
  }
});
