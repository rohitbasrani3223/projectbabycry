/* ================================================================
   CrySense — Frontend Logic (Enhanced with Emotions)
   ================================================================ */

const API = 'http://localhost:5000';

// ── DOM Refs ──────────────────────────────────────────────────────
const audioInput     = document.getElementById('audioInput');
const dropZone       = document.getElementById('dropZone');
const filePreview    = document.getElementById('filePreview');
const fileName       = document.getElementById('fileName');
const fileSize       = document.getElementById('fileSize');
const audioPlayer    = document.getElementById('audioPlayer');
const changeFile     = document.getElementById('changeFile');
const analyzeBtn     = document.getElementById('analyzeBtn');
const uploadCard     = document.getElementById('uploadCard');
const loadingCard    = document.getElementById('loadingCard');
const resultCard     = document.getElementById('resultCard');
const tryAgainBtn    = document.getElementById('tryAgainBtn');
const modelStatus    = document.getElementById('modelStatus');
const statAcc        = document.getElementById('statAcc');
const statParams     = document.getElementById('statParams');
const chartCard      = document.getElementById('chartCard');
const notTrainedNote = document.getElementById('notTrainedNote');

let selectedFile   = null;
let trainChart     = null;
let currentMetric  = 'accuracy';
let historyData    = null;
let currentResult  = null;  // store for language toggle
let currentLang    = 'en';

// ── Utility ───────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b < 1024)      return b + ' B';
  if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB';
  return (b/1024/1024).toFixed(2) + ' MB';
}
function fmtNum(n) {
  if (n >= 1_000_000) return (n/1_000_000).toFixed(2) + 'M';
  if (n >= 1_000)     return (n/1_000).toFixed(1) + 'K';
  return n;
}

// ── Language Toggle ───────────────────────────────────────────────
function switchLang(lang) {
  currentLang = lang;
  document.getElementById('langEn').classList.toggle('active', lang === 'en');
  document.getElementById('langHi').classList.toggle('active', lang === 'hi');
  if (currentResult) updateAdvice(currentResult);
}

function updateAdvice(data) {
  const advice = currentLang === 'hi' && data.advice_hi
    ? data.advice_hi
    : data.advice;
  document.getElementById('adviceText').textContent = advice ?? '—';
}

// ── Model Status & Stats ──────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res  = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    const dot  = modelStatus.querySelector('.status-dot');
    const txt  = modelStatus.querySelector('.status-text');
    dot.className = 'status-dot ' + (data.model_loaded ? 'online' : 'offline');
    txt.textContent = data.model_loaded ? 'Model Ready' : 'Model Not Trained';
  } catch {
    const dot = modelStatus.querySelector('.status-dot');
    const txt = modelStatus.querySelector('.status-text');
    dot.className = 'status-dot offline';
    txt.textContent = 'Server Offline';
  }
}

async function loadModelInfo() {
  try {
    const res  = await fetch(`${API}/model-info`, { signal: AbortSignal.timeout(4000) });
    const info = await res.json();
    if (info.status === 'ready') {
      statAcc.textContent    = info.best_val_acc != null ? info.best_val_acc + '%' : '—';
      statParams.textContent = info.total_params ? fmtNum(info.total_params) : '—';
      notTrainedNote.classList.add('hidden');
      chartCard.style.display = '';
      loadTrainingChart();
    } else {
      notTrainedNote.classList.remove('hidden');
      chartCard.style.display = 'none';
      statAcc.textContent    = '—';
      statParams.textContent = '—';
    }
  } catch {
    notTrainedNote.classList.remove('hidden');
  }
}

async function loadTrainingChart() {
  try {
    const res  = await fetch(`${API}/history`, { signal: AbortSignal.timeout(4000) });
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
          data:  historyData[metric] || [],
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(6,182,212,0.10)',
          tension: 0.4, fill: true, pointRadius: 3,
          pointBackgroundColor: '#22d3ee', borderWidth: 2,
        },
        {
          label: 'Val ' + (metric === 'accuracy' ? 'Accuracy' : 'Loss'),
          data:  historyData['val_' + metric] || [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52,211,153,0.08)',
          tension: 0.4, fill: true, pointRadius: 3,
          pointBackgroundColor: '#34d399', borderWidth: 2,
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#aaa', font: { family: 'Outfit', size: 12 } } }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#666', font: { family: 'Outfit' } },
          title: { display: true, text: 'Epoch', color: '#555', font: { family: 'Outfit' } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#666', font: { family: 'Outfit' } },
          min: metric === 'accuracy' ? 0 : undefined,
          max: metric === 'accuracy' ? 1 : undefined,
        }
      }
    }
  });
}

// Chart Tab Toggle
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
  audioPlayer.src      = URL.createObjectURL(file);
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

// ── Loading Stepper ───────────────────────────────────────────────
async function animateSteps() {
  const steps = ['step1', 'step2', 'step3'];
  for (let i = 0; i < steps.length; i++) {
    await new Promise(r => setTimeout(r, i === 0 ? 400 : 900));
    if (i > 0) {
      document.getElementById(steps[i-1]).classList.remove('active');
      document.getElementById(steps[i-1]).classList.add('done');
    }
    document.getElementById(steps[i]).classList.add('active');
  }
}

// ── Run Analysis ──────────────────────────────────────────────────
async function analyze() {
  if (!selectedFile) return;
  uploadCard.classList.add('hidden');
  resultCard.classList.add('hidden');
  loadingCard.classList.remove('hidden');

  ['step1','step2','step3'].forEach(id => {
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
  } catch(e) {
    result = { status: 'error', error: 'Could not connect to CrySense server. Make sure app.py is running.' };
  }

  await stepAnim;
  await new Promise(r => setTimeout(r, 600));
  loadingCard.classList.add('hidden');

  if (result.status === 'error' || result.error) {
    showError(result.error || 'Unknown error');
    uploadCard.classList.remove('hidden');
    return;
  }

  showResult(result);
}

analyzeBtn.addEventListener('click', analyze);
tryAgainBtn.addEventListener('click', () => {
  resultCard.classList.add('hidden');
  uploadCard.classList.remove('hidden');
});

// ── Render Result ─────────────────────────────────────────────────
function showResult(data) {
  currentResult = data;
  currentLang   = 'en';
  document.getElementById('langEn').classList.add('active');
  document.getElementById('langHi').classList.remove('active');

  // Main prediction
  document.getElementById('resultEmoji').textContent      = data.emoji ?? '👶';
  document.getElementById('resultTitle').textContent      = data.label ?? (data.prediction ?? '—').replace(/_/g,' ');
  document.getElementById('resultSubLabel').textContent   = `Detected: ${(data.prediction ?? '—').replace(/_/g,' ')}`;
  document.getElementById('resultConfidence').textContent = `Confidence: ${data.confidence ?? '—'}%`;
  document.getElementById('resultTitle').style.color      = data.color ?? '#6c63ff';

  // Advice
  updateAdvice(data);

  // Duration tip
  const durationTip = document.getElementById('durationTip');
  if (data.duration_tip) {
    durationTip.textContent = '⏱️ ' + data.duration_tip;
    durationTip.classList.remove('hidden');
  } else {
    durationTip.classList.add('hidden');
  }

  // Severity badge
  const sev    = data.severity ?? 'low';
  const colors = { low: '#22c55e', medium: '#f97316', high: '#ef4444' };
  const labels = { low: '🟢 Low Urgency', medium: '🟠 Medium Urgency', high: '🔴 High Urgency' };
  const notes  = { low: 'Attend when convenient', medium: 'Respond soon', high: 'Act immediately' };
  const badge  = document.getElementById('sevBadge');
  badge.textContent      = labels[sev] ?? sev;
  badge.style.background = (colors[sev] ?? '#888') + '22';
  badge.style.border     = `1px solid ${colors[sev] ?? '#888'}55`;
  badge.style.color      = colors[sev] ?? '#888';
  document.getElementById('sevNote').textContent = notes[sev] ?? '';

  // Quick Tips
  const tipsGrid = document.getElementById('tipsGrid');
  tipsGrid.innerHTML = '';
  const tips = data.quick_tips ?? [];
  if (tips.length > 0) {
    document.getElementById('tipsSection').classList.remove('hidden');
    tips.forEach(tip => {
      const el = document.createElement('div');
      el.className = 'tip-chip';
      el.textContent = tip;
      tipsGrid.appendChild(el);
    });
  } else {
    document.getElementById('tipsSection').classList.add('hidden');
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

  // Inferred / Secondary Emotion
  const inferredBox = document.getElementById('inferredBox');
  const inferred = data.inferred_emotion;
  if (inferred) {
    document.getElementById('inferredEmoji').textContent = inferred.emoji ?? '🔍';
    document.getElementById('inferredLabel').textContent = inferred.label ?? '—';
    document.getElementById('inferredDesc').textContent  = inferred.desc ?? '—';
    document.getElementById('inferredTip').textContent   = '💡 ' + (inferred.tip ?? '');
    inferredBox.classList.remove('hidden');
    inferredBox.style.borderColor = inferred.color ?? '#6c63ff';
  } else {
    inferredBox.classList.add('hidden');
  }

  // Probability Bars
  const probBars = document.getElementById('probBars');
  probBars.innerHTML = '';
  const colorMap = {
    belly_pain: '#ef4444',
    burping:    '#f97316',
    discomfort: '#eab308',
    hungry:     '#22c55e',
    tired:      '#6c63ff'
  };
  const probs  = data.all_probs ?? {};
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, pct]) => {
    const row   = document.createElement('div');
    row.className = 'prob-row';

    const label = document.createElement('span');
    label.className = 'prob-label';
    label.textContent = cls.replace(/_/g, ' ');

    const track = document.createElement('div');
    track.className = 'prob-track';
    const fill = document.createElement('div');
    fill.className = 'prob-fill';
    fill.style.background = colorMap[cls] ?? '#6c63ff';
    fill.style.width = '0%';
    track.appendChild(fill);

    const pctEl = document.createElement('span');
    pctEl.className = 'prob-pct';
    pctEl.textContent = pct.toFixed(1) + '%';

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(pctEl);
    probBars.appendChild(row);

    setTimeout(() => { fill.style.width = pct + '%'; }, 50);
  });

  resultCard.classList.remove('hidden');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(msg) {
  alert('❌ Error: ' + msg);
}

// ── Init ──────────────────────────────────────────────────────────
(async () => {
  await checkModelStatus();
  await loadModelInfo();

  window.addEventListener('scroll', () => {
    document.getElementById('navbar').style.background =
      window.scrollY > 20
        ? 'rgba(6,6,18,0.9)'
        : 'rgba(6,6,18,0.7)';
  });
})();
