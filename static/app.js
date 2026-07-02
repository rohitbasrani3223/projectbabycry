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
      // Extremely subtle and premium max 2 degrees tilt
      const tiltX = dy * -2.2;
      const tiltY = dx * 2.2;
      card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-2px)`;
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

  document.querySelectorAll('.reveal-up, .reveal-right, .reveal-3d-left, .reveal-3d-right, .reveal-3d-scale').forEach(el => obs.observe(el));
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

    // Get active theme colors dynamically
    const themeAccent = getComputedStyle(document.body).getPropertyValue('--accent-lt').trim() || '#22d3ee';
    const themeAccent2 = getComputedStyle(document.body).getPropertyValue('--accent2-lt').trim() || '#34d399';
    const themeGlow = getComputedStyle(document.body).getPropertyValue('--accent-glow').trim() || 'rgba(34,211,238,0.35)';

    // Gradient stroke
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, themeAccent);
    grad.addColorStop(0.5, themeAccent2);
    grad.addColorStop(1, themeAccent);
    
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = themeGlow;
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

  const themeBorder = getComputedStyle(document.body).getPropertyValue('--border').trim() || 'rgba(6,182,212,0.2)';

  // Draw flat idle line
  ctx.beginPath();
  ctx.strokeStyle = themeBorder;
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

  // Spectrogram plot display
  const specSec = document.getElementById('spectrogramSection');
  const specPlot = document.getElementById('resultPlot');
  if (data.plot_url) {
    specPlot.src = data.plot_url + '?t=' + new Date().getTime();
    specSec.classList.remove('hidden');
  } else {
    specSec.classList.add('hidden');
  }

  // Care Plan report reset
  const careSec = document.getElementById('careReportSection');
  if (careSec) {
    careSec.classList.remove('hidden');
    document.getElementById('reportGeneratorState').classList.remove('hidden');
    document.getElementById('reportLoadingState').classList.add('hidden');
    document.getElementById('reportContentState').classList.add('hidden');
    const actionsState = document.getElementById('reportActionsState');
    if (actionsState) actionsState.classList.add('hidden');
    document.getElementById('reportText').innerHTML = '';
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

// ── Theme Selection & Persistence ───────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem('crySenseTheme') || 'cyberpunk';
  setTheme(savedTheme);
}

function setTheme(themeName) {
  document.body.setAttribute('data-theme', themeName);
  localStorage.setItem('crySenseTheme', themeName);
  
  // Update active state in dropdown list
  document.querySelectorAll('.theme-opt').forEach(opt => {
    opt.classList.toggle('active', opt.dataset.theme === themeName);
  });
  
  // Close dropdown menu
  const menu = document.getElementById('themeDropdown');
  if (menu) menu.classList.remove('open');
}

function toggleThemeDropdown(event) {
  event.stopPropagation();
  const menu = document.getElementById('themeDropdown');
  if (menu) menu.classList.toggle('open');
}

// Close dropdown if user clicks outside
document.addEventListener('click', () => {
  const menu = document.getElementById('themeDropdown');
  if (menu) menu.classList.remove('open');
});

// ── Interactive Particles Simulation ──
let particleCanvasElement = null;
let particleCtx = null;
let particles = [];
const particleCount = 48;
let mousePosition = { x: null, y: null };

function initParticles() {
  particleCanvasElement = document.getElementById('particleCanvas');
  if (!particleCanvasElement) return;
  particleCtx = particleCanvasElement.getContext('2d');
  
  resizeParticleCanvas();
  window.addEventListener('resize', resizeParticleCanvas);
  
  // Track mouse movements
  window.addEventListener('mousemove', (e) => {
    mousePosition.x = e.clientX;
    mousePosition.y = e.clientY;
  });
  window.addEventListener('mouseleave', () => {
    mousePosition.x = null;
    mousePosition.y = null;
  });
  
  // Initialize nodes
  particles = [];
  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      radius: Math.random() * 2 + 1.2,
      baseRadius: Math.random() * 2 + 1.2
    });
  }
  
  animateParticles();
}

function resizeParticleCanvas() {
  if (!particleCanvasElement) return;
  particleCanvasElement.width = window.innerWidth;
  particleCanvasElement.height = window.innerHeight;
}

function animateParticles() {
  if (!particleCtx || !particleCanvasElement) return;
  requestAnimationFrame(animateParticles);
  
  particleCtx.clearRect(0, 0, particleCanvasElement.width, particleCanvasElement.height);
  
  // Fetch dynamic colors from the theme styling
  const themeAccent = getComputedStyle(document.body).getPropertyValue('--accent-lt').trim() || '#22d3ee';
  const rgbAccent = hexToRgb(themeAccent);
  
  // Draw particles
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    
    // Move
    p.x += p.vx;
    p.y += p.vy;
    
    // Boundary check
    if (p.x < 0 || p.x > particleCanvasElement.width) p.vx *= -1;
    if (p.y < 0 || p.y > particleCanvasElement.height) p.vy *= -1;
    
    // Mouse attraction/repulsion
    if (mousePosition.x !== null && mousePosition.y !== null) {
      const dx = mousePosition.x - p.x;
      const dy = mousePosition.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 180) {
        // Subtle drag attraction
        p.x += dx * 0.003;
        p.y += dy * 0.003;
        p.radius = p.baseRadius * 1.5;
      } else {
        p.radius = p.baseRadius;
      }
    }
    
    // Draw dot
    particleCtx.beginPath();
    particleCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    particleCtx.fillStyle = `rgba(${rgbAccent.r}, ${rgbAccent.g}, ${rgbAccent.b}, 0.18)`;
    particleCtx.fill();
    
    // Connect particles
    for (let j = i + 1; j < particles.length; j++) {
      const p2 = particles[j];
      const dx = p.x - p2.x;
      const dy = p.y - p2.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < 110) {
        particleCtx.beginPath();
        particleCtx.moveTo(p.x, p.y);
        particleCtx.lineTo(p2.x, p2.y);
        const alpha = (1 - dist / 110) * 0.065;
        particleCtx.strokeStyle = `rgba(${rgbAccent.r}, ${rgbAccent.g}, ${rgbAccent.b}, ${alpha})`;
        particleCtx.lineWidth = 0.8;
        particleCtx.stroke();
      }
    }
  }
}

// Utility to convert hex to rgb
function hexToRgb(hex) {
  // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
  const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
  hex = hex.replace(shorthandRegex, (m, r, g, b) => r + r + g + g + b + b);
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 34, g: 211, b: 238 }; // cyan default
}

// ── Scroll Depth Progress Indicator ──
function initScrollProgress() {
  window.addEventListener('scroll', () => {
    const scrollProgress = document.getElementById('scrollProgress');
    if (!scrollProgress) return;
    
    const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;
    scrollProgress.style.width = scrolled + '%';
  });
}

// ── Pediatric Care Plan Report Templates ──
const CARE_PLAN_TEMPLATES = {
  hungry: `
<h5>🍼 FEEDING STRATEGY & SOOTHING ACTION PLAN</h5>
<p>Based on CrySense acoustical markers, your baby is presenting a **Hunger Cry**. Rhythmic breathing gaps and sucking reflex cues verify this state.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Feed Immediately**: Offer breast or bottle. Do not wait until crying becomes more intense as it makes latching harder and causes baby to swallow air.</li>
  <li>**Rooting Checks**: Watch for baby turning head side-to-side, smacking lips, or sucking fingers. These are early cues for next feeds.</li>
  <li>**Pace Feeding**: If bottle-feeding, keep baby upright and hold bottle horizontally to prevent choking.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Track Output**: Monitor wet diapers. A well-fed infant should produce 6+ heavy wet diapers per 24 hours.</li>
  <li>**Feedings Schedule**: Newborns generally feed every 2-3 hours (8-12 times a day). Keep a log to understand your baby's routine.</li>
</ul>
  `,
  belly_pain: `
<h5>🤢 SOOTHING PLAN FOR BELLY PAIN & GAS</h5>
<p>CrySense has detected acoustic patterns indicative of **Belly Pain**. Intense, high-pitched shrieks and sudden stops point towards intestinal pressure or colic.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Bicycle Legs**: Lay baby on back and gently move their legs in a bicycling motion toward their abdomen to release trapped gas.</li>
  <li>**Tummy Massage**: Rub baby's tummy gently in slow clockwise circles to match the digestion tract.</li>
  <li>**Tummy Time**: Place baby tummy-down across your lap while supporting their head. Faint pressure helps release abdominal blocks.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Feed Position**: Hold baby slightly upright during feeds. Feed slower and burp mid-feed if they swallow air rapidly.</li>
  <li>**Warning Cues**: If the baby has a fever, green vomit, or is completely inconsolable for over 3 hours, call your pediatrician immediately.</li>
</ul>
  `,
  burping: `
<h5>💨 DIGESTIVE PLAN & BURPING TECHNIQUE</h5>
<p>Your baby is showing signs of trapped air or the need to **Burp**. Rhythmic, brief pauses with guttural cries show bubbles stuck in the throat or chest.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Over-the-Shoulder Method**: Hold baby upright against your chest with chin resting on your shoulder. Support their bottom and gently pat or rub their back in upward motions.</li>
  <li>**Sitting Upright Method**: Sit baby on your lap leaning slightly forward. Support their chest and chin with one hand (never grab the throat) and pat their back with the other.</li>
  <li>**Lap Method**: Lay baby face-down across your lap, supporting their head so it's slightly higher than their chest. Pat back gently.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Burp Frequency**: Burp your baby every 1-2 ounces during bottle feeds, or when switching breasts.</li>
  <li>**Reflux Tips**: Keep baby upright for 15-20 minutes after feedings to reduce spitting up.</li>
</ul>
  `,
  discomfort: `
<h5>😣 COMFORT AND TEMPERATURE REGULATION PLAN</h5>
<p>The prediction indicates **Discomfort**. Rhythmic, whiny cries usually mean some physical stimulus is causing irritation (wet diaper, itchy clothing, room temperature, skin chafing).</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Diaper Inspection**: Check and change their diaper immediately. Even small amounts of wetness cause sensitive skin irritation.</li>
  <li>**Temperature Check**: Feel baby's chest or back of neck (hands/feet are normally cold). If hot/sweaty, remove layers. If cold, add a swaddle layer.</li>
  <li>**Clothing Check**: Check for tags, tight zippers, or hair tourniquets wrapped around fingers or toes.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Optimal Room Temp**: Keep the nursery between 20°C and 22°C (68°F - 72°F) which is clinically ideal for safe sleep.</li>
  <li>**Chafing Care**: Apply barrier cream if skin redness or rash is observed. Use breathable cotton clothing.</li>
</ul>
  `,
  tired: `
<h5>😴 SLEEP RESTORATION & CALMING SYSTEM</h5>
<p>CrySense has identified a **Tired Cry**. Whiny, nasal sounds with dropping pitch indicate overtiredness and sensory overload.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Reduce Stimulation**: Instantly dim lights, turn off screens/music, and move to a quiet dark room.</li>
  <li>**Snug Swaddle**: Swaddle baby snugly to restrict startle reflexes (Moro reflex) and provide a womb-like safety.</li>
  <li>**Rhythmic Soothing**: Play low white noise (ocean waves or fan sound) and rock them gently with vertical movements.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Watch Sleep Cues**: Put baby down to sleep as soon as you see early yawning, eye-rubbing, or blank staring. Waiting too long makes them overtired and harder to soothe.</li>
  <li>**Routine Building**: Maintain a calm bedtime routine (warm bath, soft book, quiet lullaby) to build positive sleep cues.</li>
</ul>
  `,
  cold_hot: `
<h5>🌡️ THERMAL COMFORT & REGULATION SYSTEM</h5>
<p>The acoustics show **Thermal Discomfort** (too cold or too hot). Crying due to temperature shifts is characterized by urgent, high-intensity vocal stress.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Feel Body Core**: Feel the baby's chest, neck, or back. If sweaty or hot, remove a layer of clothes. If cold, add a cotton layers or swaddle.</li>
  <li>**Adjust Room Temp**: Ensure the nursery is set to the optimal temperature of 20°C - 22°C.</li>
  <li>**Avoid Over-wrapping**: Wrapping too tightly when hot leads to heat rashes and increases SIDS risks. Ensure comfortable airflow.</li>
</ul>

<h5>👶 CLINICAL ADVICE & LONG-TERM CARE</h5>
<ul>
  <li>**Layering Rule**: Dress baby in one additional layer than what you would wear comfortably in the same environment.</li>
  <li>**Fever Check**: If baby feels excessively hot and is lethargic, use a thermometer to check for fever. Call doctor if temperature is > 38°C (100.4°F).</li>
</ul>
  `,
  uncertain: `
<h5>👶 COMFORTING & INITIAL ASSESSMENT PLAN</h5>
<p>Acoustic signals present an **Uncertain Cry**. It is recommended to perform a step-by-step soothing checks.</p>

<h5>⏰ IMMEDIATE STEPS</h5>
<ul>
  <li>**Check Hunger First**: If it has been 2+ hours since last feed, offer milk.</li>
  <li>**Inspect Diaper**: Ensure diaper is clean and dry.</li>
  <li>**Skin-to-Skin**: Hold baby against your bare skin. Human touch naturally lowers heart rates and cortisol in infants.</li>
  <li>**Sucking Comfort**: Offer a pacifier or clean finger to suck, which activates soothing reflex pathways.</li>
</ul>
  `
};

function generateCareReport() {
  if (!currentResult) return;
  
  const genState = document.getElementById('reportGeneratorState');
  const loadState = document.getElementById('reportLoadingState');
  const contentState = document.getElementById('reportContentState');
  const reportText = document.getElementById('reportText');
  const actionsState = document.getElementById('reportActionsState');
  
  genState.classList.add('hidden');
  loadState.classList.remove('hidden');
  
  // Simulate acoustic analysis processing
  setTimeout(() => {
    loadState.classList.add('hidden');
    contentState.classList.remove('hidden');
    if (actionsState) actionsState.classList.remove('hidden');
    
    // Fetch custom advice based on prediction
    const pred = currentResult.prediction || 'uncertain';
    const textHtml = CARE_PLAN_TEMPLATES[pred] || CARE_PLAN_TEMPLATES['uncertain'];
    
    // Stream text in typewriter format
    streamHtmlText(reportText, textHtml);
  }, 1800);
}

function streamHtmlText(element, html) {
  element.innerHTML = '';
  // Stream tags to avoid breaking layout
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = html;
  const nodes = Array.from(tempDiv.childNodes);
  
  let nodeIndex = 0;
  function addNextNode() {
    if (nodeIndex >= nodes.length) return;
    
    const node = nodes[nodeIndex].cloneNode(true);
    element.appendChild(node);
    
    // Typewriter effect on text nodes
    if (node.nodeType === Node.TEXT_NODE) {
      const origText = node.textContent;
      node.textContent = '';
      let charIndex = 0;
      const interval = setInterval(() => {
        if (charIndex >= origText.length) {
          clearInterval(interval);
          nodeIndex++;
          addNextNode();
        } else {
          node.textContent += origText[charIndex];
          charIndex++;
        }
      }, 3);
    } else {
      // Tags render instantly
      nodeIndex++;
      setTimeout(addNextNode, 40);
    }
  }
  
  addNextNode();
}

// ── Chat Copilot Widget Handler ──
function toggleChatWidget(event) {
  if (event) event.stopPropagation();
  const container = document.getElementById('chatContainer');
  if (container) {
    container.classList.toggle('open');
    if (container.classList.contains('open')) {
      document.getElementById('chatInput').focus();
    }
  }
}

const CHAT_ANSWERS = {
  feed: "For newborn babies, it is recommended to feed on demand, usually every 2 to 3 hours. Watch out for early hunger signals like rooting (turning head to find the breast), hand sucking, or lip smacking.",
  bottle: "Ensure the bottle nipple flow is appropriate for your baby's age. Keep baby's head elevated higher than the stomach to reduce gas, and burp every 1-2 ounces.",
  gas: "To soothe baby gas, lay baby on their back and gently bike their legs in circles. Massage their tummy clockwise or try holding them in the 'colic carry' (chest down over your forearm).",
  burp: "Try patting the baby's back gently with a cupped hand. Key positions are upright on your shoulder, sitting on your lap while supporting their chin (not throat), or facedown across your lap.",
  diaper: "Check diaper every 2-3 hours. Prolonged wetness can cause diaper rash. Apply diaper paste if skin looks red or irritated.",
  sleep: "Newborns sleep 16-18 hours per day, but in short chunks. Put baby down when they show signs of drowsiness (yawning, eye rubbing, staring). Keep room cool, dark, and use white noise.",
  fever: "In babies under 3 months, a rectal temperature of 38°C (100.4°F) or higher is considered a fever and requires immediate consultation with a pediatrician. Do not administer medication without a doctor's advice.",
  soothe: "Try the 5 S's technique: Swaddle, Side/Stomach position in your arms, Shush (white noise), Swing (rhythmic rocking), and Suck (pacifier or clean finger)."
};

function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const messages = document.getElementById('chatMessages');
  if (!input || !messages || !input.value.trim()) return;
  
  const text = input.value.trim();
  input.value = '';
  
  // Append user message
  const userMsg = document.createElement('div');
  userMsg.className = 'chat-msg user';
  userMsg.innerHTML = `<p>${text}</p>`;
  messages.appendChild(userMsg);
  messages.scrollTop = messages.scrollHeight;
  
  // Show typing indicator
  const typingInd = document.createElement('div');
  typingInd.className = 'typing-indicator';
  typingInd.id = 'typingIndicator';
  typingInd.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  messages.appendChild(typingInd);
  messages.scrollTop = messages.scrollHeight;
  
  // Parse response
  setTimeout(() => {
    // Remove typing indicator
    const ind = document.getElementById('typingIndicator');
    if (ind) ind.remove();
    
    // Find answer
    let answer = "I'm here to help! Could you ask a question about baby feeding, sleep, burping, gas relief, or how to soothe your crying baby?";
    const textLower = text.toLowerCase();
    
    if (textLower.includes('feed') || textLower.includes('hungry') || textLower.includes('milk') || textLower.includes('breast')) {
      answer = CHAT_ANSWERS.feed;
    } else if (textLower.includes('bottle') || textLower.includes('formula')) {
      answer = CHAT_ANSWERS.bottle;
    } else if (textLower.includes('gas') || textLower.includes('colic') || textLower.includes('stomach') || textLower.includes('tummy') || textLower.includes('pain')) {
      answer = CHAT_ANSWERS.gas;
    } else if (textLower.includes('burp')) {
      answer = CHAT_ANSWERS.burp;
    } else if (textLower.includes('diaper') || textLower.includes('rash') || textLower.includes('wet')) {
      answer = CHAT_ANSWERS.diaper;
    } else if (textLower.includes('sleep') || textLower.includes('tired') || textLower.includes('nap') || textLower.includes('night')) {
      answer = CHAT_ANSWERS.sleep;
    } else if (textLower.includes('fever') || textLower.includes('temperature') || textLower.includes('hot') || textLower.includes('sick')) {
      answer = CHAT_ANSWERS.fever;
    } else if (textLower.includes('soothe') || textLower.includes('calm') || textLower.includes('cry') || textLower.includes('stop')) {
      answer = CHAT_ANSWERS.soothe;
    }
    
    // Append bot response
    const botMsg = document.createElement('div');
    botMsg.className = 'chat-msg bot';
    messages.appendChild(botMsg);
    
    // Stream bot response character by character
    let charIndex = 0;
    const interval = setInterval(() => {
      if (charIndex >= answer.length) {
        clearInterval(interval);
      } else {
        botMsg.innerHTML += answer[charIndex];
        charIndex++;
        messages.scrollTop = messages.scrollHeight;
      }
    }, 12);
  }, 1000);
}

function handleChatKey(event) {
  if (event.key === 'Enter') {
    sendChatMessage();
  }
}

// ── Init ──────────────────────────────────────────────────────────
(async () => {
  initTheme();
  initScrollReveal();
  initScrollProgress();
  initParticles();
  setTimeout(init3DTilt, 500); // after DOM settles
  initIdleWave();

  await checkModelStatus();
  await loadModelInfo();

  // Trigger reveal for hero (above fold)
  document.querySelectorAll('.reveal-up, .reveal-right, .reveal-3d-left, .reveal-3d-right, .reveal-3d-scale').forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight) el.classList.add('revealed');
  });
})();

function downloadCareReport() {
  if (!currentResult) return;
  const pred = currentResult.prediction || 'uncertain';
  const html = CARE_PLAN_TEMPLATES[pred] || CARE_PLAN_TEMPLATES['uncertain'];
  
  let text = html
    .replace(/<h5>(.*?)<\/h5>/g, '\n=== $1 ===\n')
    .replace(/<li>\*\*(.*?)\*\*:(.*?)<\/li>/g, '* $1: $2')
    .replace(/<li>(.*?)<\/li>/g, '* $1')
    .replace(/<p>(.*?)<\/p>/g, '\n$1\n')
    .replace(/<ul.*?>/g, '')
    .replace(/<\/ul>/g, '')
    .replace(/\*\*/g, '')
    .trim();

  const timestamp = new Date().toLocaleString();
  const label = currentResult.label || pred.toUpperCase();
  const confidence = currentResult.confidence || 0;
  const severity = currentResult.severity || 'low';
  
  const reportHeader = `==================================================
CRYSENSE CLINICAL CARE REPORT & ACTION PLAN
Generated: ${timestamp}
==================================================

[PATIENT DETAILS]
- Condition: ${label}
- Confidence: ${confidence}%
- Severity Level: ${severity.toUpperCase()}

[DETAILED ANALYSIS & RECOMMENDATIONS]
${text}

==================================================
DISCLAIMER: This report is generated by CrySense based on acoustic baby cry markers. It is for informational and educational support only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your pediatrician or other qualified health provider with any questions you may have regarding a medical condition.
==================================================`;

  const blob = new Blob([reportHeader], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `CrySense_Care_Plan_${pred}_${new Date().toISOString().slice(0,10)}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function printCareReport() {
  if (!currentResult) return;
  const pred = currentResult.prediction || 'uncertain';
  const label = currentResult.label || pred.toUpperCase();
  const confidence = currentResult.confidence || 0;
  const severity = currentResult.severity || 'low';
  const emoji = currentResult.emoji || '👶';
  const color = currentResult.color || '#22c55e';
  const advice = currentLang === 'hi' && currentResult.advice_hi ? currentResult.advice_hi : currentResult.advice;
  const timestamp = new Date().toLocaleString();
  const plotUrl = currentResult.plot_url ? `${window.location.origin}${currentResult.plot_url}?t=${new Date().getTime()}` : '';

  // Get generated care plan from screen if present, else fall back to template
  const reportTextEl = document.getElementById('reportText');
  let carePlanHtml = reportTextEl ? reportTextEl.innerHTML : '';
  if (!carePlanHtml || carePlanHtml.trim() === '' || carePlanHtml.includes('Generating')) {
    carePlanHtml = CARE_PLAN_TEMPLATES[pred] || CARE_PLAN_TEMPLATES['uncertain'];
  }

  // Class mapping colors
  const colorMap = {
    belly_pain: '#ef4444',
    burping: '#f97316',
    discomfort: '#eab308',
    hungry: '#22c55e',
    tired: '#6c63ff',
    cold_hot: '#06b6d4'
  };

  // Build probabilities bar chart HTML
  const probs = currentResult.all_probs ?? {};
  const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  let probHtml = '<div class="prob-section">';
  sortedProbs.forEach(([cls, pct]) => {
    const activeColor = colorMap[cls] ?? '#0284c7';
    probHtml += `
      <div class="prob-row">
        <span class="prob-label">${cls.replace(/_/g, ' ').toUpperCase()}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pct}%; background-color: ${activeColor};"></div>
        </div>
        <span class="prob-pct">${pct.toFixed(1)}%</span>
      </div>
    `;
  });
  probHtml += '</div>';

  const printWindow = window.open('', '_blank', 'width=850,height=850');
  printWindow.document.write(`
    <html>
      <head>
        <title>CrySense Clinical Care Report - ${label}</title>
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
          body {
            font-family: 'Outfit', sans-serif;
            color: #0f172a;
            padding: 30px;
            line-height: 1.5;
            background: #ffffff;
            margin: 0;
          }
          .no-print-bar {
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-radius: 8px;
          }
          .print-btn {
            background: #0284c7;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
          }
          .print-btn:hover {
            background: #0369a1;
          }
          .close-btn {
            background: #e2e8f0;
            color: #334155;
            border: none;
            padding: 10px 15px;
            font-size: 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
          }
          .report-container {
            border: 1px solid #e2e8f0;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
            max-width: 800px;
            margin: 0 auto;
          }
          .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 25px;
          }
          .header-title h1 {
            margin: 0;
            color: #0f172a;
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.02em;
          }
          .header-title p {
            margin: 5px 0 0 0;
            color: #64748b;
            font-size: 14px;
            font-weight: 500;
          }
          .logo {
            font-size: 28px;
          }
          .meta-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            background: #f8fafc;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border: 1px solid #f1f5f9;
          }
          .meta-item {
            font-size: 14px;
            color: #334155;
          }
          .meta-item strong {
            color: #0f172a;
          }
          
          .section-title {
            color: #0f172a;
            font-size: 15px;
            font-weight: 700;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 1.5px solid #cbd5e1;
            padding-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }

          /* Prediction block styling */
          .prediction-card {
            border: 1.5px solid ${color}44;
            background: ${color}08;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 20px;
          }
          .prediction-emoji {
            font-size: 45px;
            background: #ffffff;
            padding: 10px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgb(0 0 0 / 0.02);
          }
          .prediction-info {
            flex-grow: 1;
          }
          .prediction-name {
            font-size: 22px;
            font-weight: 700;
            color: ${color};
            margin: 0 0 5px 0;
          }
          .prediction-desc {
            font-size: 14px;
            color: #475569;
            margin: 0;
          }
          .urgency-badge {
            display: inline-block;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 700;
            border-radius: 20px;
            text-transform: uppercase;
            margin-top: 8px;
            background-color: ${color}22;
            color: ${color};
            border: 1px solid ${color}55;
          }

          /* Scientific Plot */
          .plot-container {
            text-align: center;
            margin-bottom: 25px;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            background: #fafafa;
          }
          .plot-img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
          }

          /* Probabilities grid */
          .prob-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 25px;
          }
          .prob-row {
            display: flex;
            align-items: center;
            font-size: 13px;
          }
          .prob-label {
            width: 140px;
            font-weight: 600;
            color: #334155;
          }
          .prob-track {
            flex-grow: 1;
            background: #e2e8f0;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin-right: 15px;
          }
          .prob-fill {
            height: 100%;
            border-radius: 5px;
          }
          .prob-pct {
            width: 50px;
            text-align: right;
            font-weight: 700;
            color: #0f172a;
          }

          /* Care plan details */
          .care-plan h5 {
            color: #0284c7;
            font-size: 14px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 10px;
            text-transform: uppercase;
          }
          .care-plan ul {
            padding-left: 20px;
            margin: 10px 0;
          }
          .care-plan li {
            margin-bottom: 8px;
            font-size: 14px;
            color: #334155;
          }

          .disclaimer {
            margin-top: 40px;
            font-size: 11px;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
            padding-top: 15px;
            text-align: justify;
            line-height: 1.4;
          }

          @media print {
            .no-print { display: none !important; }
            body { padding: 0; }
            .report-container {
              border: none;
              padding: 0;
              box-shadow: none;
              max-width: 100%;
            }
          }
        </style>
      </head>
      <body>
        <div class="no-print-bar no-print">
          <div>
            <span style="font-weight: 600; color: #475569;">CrySense Report Preview</span>
          </div>
          <div style="display: flex; gap: 10px;">
            <button class="close-btn" onclick="window.close()">Close</button>
            <button class="print-btn" onclick="window.print()">Print / Save PDF</button>
          </div>
        </div>
        
        <div class="report-container">
          <div class="header">
            <div class="header-title">
              <h1>🍼 CRYSENSE CLINICAL CARE REPORT</h1>
              <p>Acoustic Cry Signature Analysis & Action Plan</p>
            </div>
            <div class="logo">🧬</div>
          </div>
          
          <div class="meta-grid">
            <div class="meta-item"><strong>Patient Reference:</strong> Infant Cry Sample</div>
            <div class="meta-item"><strong>Analysis Timestamp:</strong> ${timestamp}</div>
            <div class="meta-item"><strong>Classification Mode:</strong> Ensemble Neural Net</div>
            <div class="meta-item"><strong>Clinical Guidance:</strong> Pediatric Standard v17</div>
          </div>

          <div class="section-title">Acoustic Classification Results</div>
          
          <div class="prediction-card">
            <div class="prediction-emoji">${emoji}</div>
            <div class="prediction-info">
              <div class="prediction-name">${label.toUpperCase()}</div>
              <div class="prediction-desc">${advice}</div>
              <span class="urgency-badge">${severity.toUpperCase()} URGENCY</span>
            </div>
            <div style="text-align: right; min-width: 100px;">
              <div style="font-size: 28px; font-weight: 800; color: ${color};">${confidence}%</div>
              <div style="font-size: 11px; color: #64748b; font-weight: 600;">CONFIDENCE</div>
            </div>
          </div>

          ${plotUrl ? `
            <div class="section-title">Scientific Audio Analysis</div>
            <div class="plot-container">
              <img class="plot-img" src="${plotUrl}" alt="Acoustic Waveform and MFCC Spectrogram Analysis" />
              <div style="font-size: 11px; color: #64748b; margin-top: 8px; font-weight: 500;">
                Figure 1: Extracted raw audio waveform (top) and Mel-Frequency Cepstral Coefficients (MFCC) spectrogram (bottom)
              </div>
            </div>
          ` : ''}

          <div class="section-title">All Classification Probabilities</div>
          ${probHtml}

          <div class="section-title">Pediatric Care Plan</div>
          <div class="care-plan">
            ${carePlanHtml}
          </div>

          <div class="disclaimer">
            <strong>Clinical Disclaimer:</strong> This report is generated automatically by CrySense based on machine learning analysis of acoustic crying patterns. It is intended solely for informational, educational, and parent-supportive reference. It does NOT constitute medical advice, diagnosis, or clinical treatment. Always consult with a registered pediatrician or medical professional for infant health concerns.
          </div>
        </div>
      </body>
    </html>
  `);
  printWindow.document.close();
}

// Expose globals needed by HTML onclick
window.switchMode = switchMode;
window.switchLang = switchLang;
window.toggleMic = toggleMic;
window.toggleThemeDropdown = toggleThemeDropdown;
window.setThemeOption = setTheme;
window.generateCareReport = generateCareReport;
window.downloadCareReport = downloadCareReport;
window.printCareReport = printCareReport;
window.toggleChatWidget = toggleChatWidget;
window.sendChatMessage = sendChatMessage;
window.handleChatKey = handleChatKey;