/* ================================================================
  CrySense — Frontend Logic
  Upload + Live Mic + Scroll Reveal + 3D Tilt + Cursor Glow
  ================================================================ */

const API = 'http://localhost:5000';

// ── DOM Refs ──────────────────────────────────────────────────────
const audioInput = document.getElementById('audioInput');
const dropZone = document.getElementById('dropZone');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const audioPlayer = document.getElementById('audioPlayer');
const changeFile = document.getElementById('changeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadCard = document.querySelector('.upload-card');
const loadingCard = document.getElementById('loadingCard');
const resultCard = document.getElementById('resultCard');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const modelStatus = document.getElementById('modelStatus');
const statAcc = document.getElementById('statAcc');
const statParams = document.getElementById('statParams');
const chartCard = document.getElementById('chartCard');
const notTrainedNote = document.getElementById('notTrainedNote');

let selectedFile = null;
let trainChart = null;
let currentMetric = 'accuracy';
let historyData = null;
let currentResult = null;
let currentLang = 'en';

// ── Utility ───────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1024 / 1024).toFixed(2) + ' MB';
}
function fmtNum(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n;
}

// ── Cursor Glow ───────────────────────────────────────────────────
const cursorGlow = document.getElementById('cursorGlow');
document.addEventListener('mousemove', e => {
  if (cursorGlow) {
    cursorGlow.style.left = e.clientX + 'px';
    cursorGlow.style.top = e.clientY + 'px';
  }
});
document.querySelectorAll('button, a, .drop-zone, .type-card, .step-card, .mode-tab, .toggle-wrap, label').forEach(el => {
  el.addEventListener('mouseenter', () => cursorGlow && cursorGlow.classList.add('active'));
  el.addEventListener('mouseleave', () => cursorGlow && cursorGlow.classList.remove('active'));
});

// ── 3D Card Tilt ─────────────────────────────────────────────────
function init3DTilt() {
  document.querySelectorAll('.glass-card, .stat-card, .step-card, .type-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (rect.width / 2);
      const dy = (e.clientY - cy) / (rect.height / 2);
      const tiltX = dy * -6;
      const tiltY = dx * 6;
      card.style.transform = `perspective(800px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-4px)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

// ── Scroll Reveal ─────────────────────────────────────────────────
function initScrollReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.reveal-up, .reveal-right').forEach(el => obs.observe(el));
}

// ── Navbar Scroll ─────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (nav) nav.classList.toggle('scrolled', window.scrollY > 30);
});

// ── Mode Switch (Upload / Live) ───────────────────────────────────
function switchMode(mode) {
  document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
  document.getElementById('tabLive').classList.toggle('active', mode === 'live');
  document.getElementById('uploadPanel').classList.toggle('hidden', mode !== 'upload');
  document.getElementById('livePanel').classList.toggle('hidden', mode !== 'live');
  loadingCard.classList.add('hidden');
  resultCard.classList.add('hidden');

  if (mode === 'live') {
    document.getElementById('resultSource').textContent = '🎤 Live Mic Analysis';
  } else {
    document.getElementById('resultSource').textContent = '📁 File Analysis';
  }
}

// ── Language Toggle ───────────────────────────────────────────────
function switchLang(lang) {
  currentLang = lang;
  document.getElementById('langEn').classList.toggle('active', lang === 'en');
  document.getElementById('langHi').classList.toggle('active', lang === 'hi');
  if (currentResult) updateAdvice(currentResult);
}

function updateAdvice(data) {
  const advice = currentLang === 'hi' && data.advice_hi ? data.advice_hi : data.advice;
  document.getElementById('adviceText').textContent = advice ?? '—';
}

// ── Model Status & Stats ──────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    const dot = modelStatus.querySelector('.status-dot');
    const txt = modelStatus.querySelector('.status-text');
    dot.className = 'status-dot ' + (data.model_loaded ? 'online' : 'offline');
    txt.textContent = data.model_loaded ? 'Model Ready' : 'Not Trained';
  } catch {
    const dot = modelStatus.querySelector('.status-dot');
    const txt = modelStatus.querySelector('.status-text');
    dot.className = 'status-dot offline';
    txt.textContent = 'Server Offline';
  }
}

async function loadModelInfo() {
  try {
    const res = await fetch(`${API}/model-info`, { signal: AbortSignal.timeout(4000) });
    const info = await res.json();
    if (info.status === 'ready') {
      statAcc.textContent = info.best_val_acc != null ? info.best_val_acc + '%' : '—';
      statParams.textContent = info.total_params ? fmtNum(info.total_params) : '—';
      notTrainedNote.classList.add('hidden');
      chartCard.style.display = '';
      loadTrainingChart();
    } else {
      notTrainedNote.classList.remove('hidden');
      chartCard.style.display = 'none';
    }
  } catch {
    notTrainedNote.classList.remove('hidden');
  }
}

async function loadTrainingChart() {
  try {
    const res = await fetch(`${API}/history`, { signal: AbortSignal.timeout(4000) });
    historyData = await res.json();
    renderChart('accuracy');
  } catch { /* ignore */ }
}

function renderChart(metric) {
  const ctx = document.getElementById('trainChart').getContext('2d');
  const epochs = Array.from({ length: historyData[metric]?.length ?? 0 }, (_, i) => i + 1);
  if (trainChart) trainChart.destroy();
  trainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Train ' + (metric === 'accuracy' ? 'Accuracy' : 'Loss'),
          data: historyData[metric] || [],
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(6,182,212,0.08)',
          tension: 0.4, fill: true, pointRadius: 2,
          pointBackgroundColor: '#22d3ee', borderWidth: 2,
        },
        {
          label: 'Val ' + (metric === 'accuracy' ? 'Accuracy' : 'Loss'),
          data: historyData['val_' + metric] || [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52,211,153,0.06)',
          tension: 0.4, fill: true, pointRadius: 2,
          pointBackgroundColor: '#34d399', borderWidth: 2,
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#64b5c9', font: { family: 'DM Sans', size: 12 } } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#3a7a8a', font: { family: 'DM Sans' } },
          title: { display: true, text: 'Epoch', color: '#3a7a8a', font: { family: 'DM Sans' } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#3a7a8a', font: { family: 'DM Sans' } },
          min: metric === 'accuracy' ? 0 : undefined,
          max: metric === 'accuracy' ? 1 : undefined,
        }
      }
    }
  });
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMetric = btn.dataset.metric;
    if (historyData) renderChart(currentMetric);
  });
});

// ── File Handling ─────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = fmtBytes(file.size);
  audioPlayer.src = URL.createObjectURL(file);
  dropZone.classList.add('hidden');
  filePreview.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

audioInput.addEventListener('change', e => handleFile(e.target.files[0]));

changeFile.addEventListener('click', () => {
  selectedFile = null;
  audioInput.value = '';
  filePreview.classList.add('hidden');
  dropZone.classList.remove('hidden');
  analyzeBtn.disabled = true;
  resultCard.classList.add('hidden');
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => audioInput.click());

// ── Loading Steps ─────────────────────────────────────────────────
async function animateSteps() {
  const steps = ['step1', 'step2', 'step3'];
  for (let i = 0; i < steps.length; i++) {
    await new Promise(r => setTimeout(r, i === 0 ? 350 : 850));
    if (i > 0) {
      document.getElementById(steps[i - 1]).classList.remove('active');
      document.getElementById(steps[i - 1]).classList.add('done');
    }
    document.getElementById(steps[i]).classList.add('active');
  }
}

// ── Upload Analyze ────────────────────────────────────────────────
async function analyze() {
  if (!selectedFile) return;
  if (uploadCard) uploadCard.classList.add('hidden');
  resultCard.classList.add('hidden');
  loadingCard.classList.remove('hidden');

  ['step1', 'step2', 'step3'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });

  const stepAnim = animateSteps();
  const fd = new FormData();
  fd.append('audio', selectedFile);

  let result;
  try {
    const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
    result = await res.json();
  } catch {
    result = { status: 'error', error: 'Cannot connect to CrySense server. Make sure app.py is running.' };
  }

  await stepAnim;
  await new Promise(r => setTimeout(r, 500));
  loadingCard.classList.add('hidden');

  if (result.status === 'error' || result.error) {
    showError(result.error || 'Unknown error');
    if (uploadCard) uploadCard.classList.remove('hidden');
    return;
  }

  document.getElementById('resultSource').textContent = '📁 File Analysis';
  showResult(result);
}

analyzeBtn.addEventListener('click', analyze);
tryAgainBtn.addEventListener('click', () => {
  resultCard.classList.add('hidden');
  const activeMode = document.getElementById('tabUpload').classList.contains('active') ? 'upload' : 'live';
  if (activeMode === 'upload' && uploadCard) uploadCard.classList.remove('hidden');
});

// ── ──────────────────────────────────────────────────────────────
//    LIVE MIC DETECTION  (4s one-shot, then auto-analyze)
// ── ──────────────────────────────────────────────────────────────
const REC_DURATION_MS = 4500;   // record exactly 4.5 seconds

let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioCtx = null;
let analyserNode = null;
let waveAnimFrame = null;
let recordingTimer = null;
let autoStopTimer = null;
let elapsedSeconds = 0;

const micBtn = document.getElementById('micBtn');
const micLabel = document.getElementById('micLabel');
const liveTimer = document.getElementById('liveTimer');
const timerText = document.getElementById('timerText');
const liveStatus = document.getElementById('liveStatus');
const liveWave = document.getElementById('liveWaveCanvas');

function toggleMic() {
  if (!isRecording) startMic();
  else stopMicEarly();
}

async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch {
    alert('\u274C Microphone access denied. Please allow microphone permission and try again.');
    return;
  }

  isRecording = true;
  audioChunks = [];
  elapsedSeconds = 0;

  // Web Audio for waveform
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 512;
  analyserNode.smoothingTimeConstant = 0.8;   // smooth waveform
  const src = audioCtx.createMediaStreamSource(mediaStream);
  src.connect(analyserNode);
  drawLiveWave();

  // MediaRecorder — collect full chunks
  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: getSupportedMime() });
  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = () => sendLiveChunk();
  mediaRecorder.start(250);

  // UI
  micBtn.classList.add('recording');
  micBtn.querySelector('.mic-icon').textContent = '\u23F9\uFE0F';
  micLabel.textContent = 'Recording 4s\u2026 tap to cancel';
  liveTimer.classList.remove('hidden');
  timerText.textContent = 'Recording: 0s';
  liveStatus.querySelector('.live-status-title').textContent = '\uD83C\uDFA4 Listening\u2026';
  liveStatus.querySelector('.live-status-sub').textContent = 'Will auto-analyze after 4 seconds.';

  // Countdown timer
  recordingTimer = setInterval(() => {
    elapsedSeconds++;
    const remaining = Math.max(0, Math.round(REC_DURATION_MS / 1000) - elapsedSeconds);
    timerText.textContent = `Recording: ${elapsedSeconds}s (${remaining}s left)`;
  }, 1000);

  // Auto-stop after REC_DURATION_MS
  autoStopTimer = setTimeout(() => stopMic(), REC_DURATION_MS);
}

function stopMic() {
  if (!isRecording) return;
  _teardownMic();
  liveStatus.querySelector('.live-status-title').textContent = '\u23F3 Analyzing\u2026';
  liveStatus.querySelector('.live-status-sub').textContent = 'Processing the 4s clip.';
}

function stopMicEarly() {
  if (!isRecording) return;
  clearTimeout(autoStopTimer);
  _teardownMic();
  liveStatus.querySelector('.live-status-title').textContent = '\u23F3 Analyzing early clip\u2026';
  liveStatus.querySelector('.live-status-sub').textContent = 'Processing whatever was recorded.';
}

function _teardownMic() {
  isRecording = false;
  clearInterval(recordingTimer);
  clearTimeout(autoStopTimer);

  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

  if (waveAnimFrame) { cancelAnimationFrame(waveAnimFrame); waveAnimFrame = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  clearLiveWave();

  liveTimer.classList.add('hidden');
  elapsedSeconds = 0;
  micBtn.classList.remove('recording');
  micBtn.querySelector('.mic-icon').textContent = '\uD83C\uDFA4';
  micLabel.textContent = 'Tap to Start Listening';
}

function getSupportedMime() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

async function sendLiveChunk() {
  if (audioChunks.length === 0) {
    liveStatus.querySelector('.live-status-title').textContent = '\u26A0\uFE0F No audio captured';
    liveStatus.querySelector('.live-status-sub').textContent = 'Try again and speak clearly near the mic.';
    return;
  }

  const mime = getSupportedMime() || 'audio/webm';
  const blob = new Blob(audioChunks, { type: mime });
  audioChunks = [];

  loadingCard.classList.remove('hidden');
  resultCard.classList.add('hidden');
  ['step1', 'step2', 'step3'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });
  const stepAnim = animateSteps();

  const fd = new FormData();
  fd.append('audio', blob, 'live_audio.wav');

  let result;
  try {
    const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
    result = await res.json();
  } catch {
    result = { status: 'error', error: 'Cannot connect to CrySense server.' };
  }

  await stepAnim;
  await new Promise(r => setTimeout(r, 400));
  loadingCard.classList.add('hidden');

  if (result.status === 'error' || result.error) {
    liveStatus.querySelector('.live-status-title').textContent = '\u26A0\uFE0F ' + (result.error || 'Error');
    liveStatus.querySelector('.live-status-sub').textContent = 'Check that the server is running.';
    return;
  }

  document.getElementById('resultSource').textContent = '\uD83C\uDFA4 Live Mic Analysis';
  showResult(result);
  liveStatus.querySelector('.live-status-title').textContent = '\u2705 Done! Tap mic to analyze again.';
  liveStatus.querySelector('.live-status-sub').textContent = 'Each tap = fresh 4-second clip.';
}

// ── Live Waveform (stable, smoothed) ─────────────────────────────
function drawLiveWave() {
  if (!analyserNode || !liveWave) return;
  const ctx = liveWave.getContext('2d');
  const W = liveWave.width;
  const H = liveWave.height;
  const buf = new Uint8Array(analyserNode.frequencyBinCount);
  let prevY = new Float32Array(buf.length).fill(H / 2);

  function draw() {
    waveAnimFrame = requestAnimationFrame(draw);
    analyserNode.getByteTimeDomainData(buf);

    ctx.clearRect(0, 0, W, H);

    // Gradient stroke
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#22d3ee');
    grad.addColorStop(0.5, '#34d399');
    grad.addColorStop(1, '#22d3ee');
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(34,211,238,0.35)';
    ctx.shadowBlur = 10;

    ctx.beginPath();
    const sliceW = W / buf.length;
    for (let i = 0; i < buf.length; i++) {
      const rawY = ((buf[i] / 128.0) * H) / 2;
      const y = prevY[i] * 0.55 + rawY * 0.45;  // smooth
      prevY[i] = y;
      const x = i * sliceW;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H / 2);
    ctx.stroke();
  }
  draw();
}

function clearLiveWave() {
  if (!liveWave) return;
  const ctx = liveWave.getContext('2d');
  ctx.clearRect(0, 0, liveWave.width, liveWave.height);

  // Draw flat idle line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(6,182,212,0.2)';
  ctx.lineWidth = 1.5;
  ctx.moveTo(0, liveWave.height / 2);
  ctx.lineTo(liveWave.width, liveWave.height / 2);
  ctx.stroke();
}

// ── Show Result ───────────────────────────────────────────────────
function showResult(data) {
  currentResult = data;
  currentLang = 'en';
  document.getElementById('langEn').classList.add('active');
  document.getElementById('langHi').classList.remove('active');

  document.getElementById('resultEmoji').textContent = data.emoji ?? '👶';
  document.getElementById('resultTitle').textContent = data.label ?? (data.prediction ?? '—').replace(/_/g, ' ');
  document.getElementById('resultSubLabel').textContent = `Detected: ${(data.prediction ?? '—').replace(/_/g, ' ')}`;
  document.getElementById('resultTitle').style.color = data.color ?? '#22d3ee';

  // Confidence bar
  const conf = data.confidence ?? 0;
  document.getElementById('resultConfidence').textContent = `${conf}%`;
  setTimeout(() => {
    const bar = document.getElementById('confBar');
    if (bar) bar.style.width = Math.min(conf, 100) + '%';
  }, 100);

  updateAdvice(data);

  // Duration tip
  const dt = document.getElementById('durationTip');
  if (data.duration_tip) { dt.textContent = '⏱️ ' + data.duration_tip; }
  else { dt.textContent = ''; }

  // Severity
  const sev = data.severity ?? 'low';
  const colors = { low: '#22c55e', medium: '#f97316', high: '#ef4444' };
  const labels = { low: '🟢 Low Urgency', medium: '🟠 Medium Urgency', high: '🔴 High Urgency' };
  const notes = { low: 'Attend when convenient', medium: 'Respond soon', high: 'Act immediately' };
  const badge = document.getElementById('sevBadge');
  badge.textContent = labels[sev] ?? sev;
  badge.style.background = (colors[sev] ?? '#888') + '22';
  badge.style.border = `1px solid ${colors[sev] ?? '#888'}55`;
  badge.style.color = colors[sev] ?? '#888';
  document.getElementById('sevNote').textContent = notes[sev] ?? '';

  // Quick Tips
  const tipsGrid = document.getElementById('tipsGrid');
  tipsGrid.innerHTML = '';
  const tips = data.quick_tips ?? [];
  const tipsSec = document.getElementById('tipsSection');
  if (tips.length > 0) {
    tipsSec.classList.remove('hidden');
    tips.forEach(tip => {
      const el = document.createElement('div');
      el.className = 'tip-chip';
      el.textContent = tip;
      tipsGrid.appendChild(el);
    });
  } else {
    tipsSec.classList.add('hidden');
  }

  // Doctor Warning
  const doctorBox = document.getElementById('doctorBox');
  const doctorText = document.getElementById('doctorText');
  if (data.doctor_warning) {
    doctorText.textContent = data.doctor_warning;
    doctorBox.classList.remove('hidden');
  } else {
    doctorBox.classList.add('hidden');
  }

  // Inferred Emotion
  const inferredBox = document.getElementById('inferredBox');
  const inferred = data.inferred_emotion;
  if (inferred) {
    document.getElementById('inferredEmoji').textContent = inferred.emoji ?? '🔍';
    document.getElementById('inferredLabel').textContent = inferred.label ?? '—';
    document.getElementById('inferredDesc').textContent = inferred.desc ?? '—';
    document.getElementById('inferredTip').textContent = '💡 ' + (inferred.tip ?? '');
    inferredBox.classList.remove('hidden');
    inferredBox.style.borderColor = inferred.color ?? '#22d3ee';
  } else {
    inferredBox.classList.add('hidden');
  }

  // Probability Bars
  const probBars = document.getElementById('probBars');
  probBars.innerHTML = '';
  const colorMap = {
    belly_pain: '#ef4444',
    burping: '#f97316',
    discomfort: '#eab308',
    hungry: '#22c55e',
    tired: '#6c63ff'
  };
  const probs = data.all_probs ?? {};
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, pct]) => {
    const row = document.createElement('div');
    row.className = 'prob-row';

    const label = document.createElement('span');
    label.className = 'prob-label';
    label.textContent = cls.replace(/_/g, ' ');

    const track = document.createElement('div');
    track.className = 'prob-track';
    const fill = document.createElement('div');
    fill.className = 'prob-fill';
    fill.style.background = colorMap[cls] ?? '#22d3ee';
    fill.style.width = '0%';
    track.appendChild(fill);

    const pctEl = document.createElement('span');
    pctEl.className = 'prob-pct';
    pctEl.textContent = pct.toFixed(1) + '%';

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pctEl);
    probBars.appendChild(row);

    setTimeout(() => { fill.style.width = pct + '%'; }, 60);
  });

  resultCard.classList.remove('hidden');
  setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function showError(msg) {
  alert('❌ Error: ' + msg);
}

// ── Init Idle Waveform ────────────────────────────────────────────
function initIdleWave() {
  if (!liveWave) return;
  clearLiveWave();
}

// ── Init ──────────────────────────────────────────────────────────
(async () => {
  initScrollReveal();
  setTimeout(init3DTilt, 500); // after DOM settles
  initIdleWave();

  await checkModelStatus();
  await loadModelInfo();

  // Trigger reveal for hero (above fold)
  document.querySelectorAll('.reveal-up, .reveal-right').forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight) el.classList.add('revealed');
  });
})();

// Expose globals needed by HTML onclick
window.switchMode = switchMode;
window.switchLang = switchLang;
window.toggleMic = toggleMic;