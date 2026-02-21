/**
 * Marjapussi AI — Frontend App
 * Connects via WebSocket to ui_server.py, renders game state,
 * handles human takeover and "Proceed move" button.
 */

const WS_URL = 'ws://localhost:8765/ws';

// ── Card data ──────────────────────────────────────────────────────────────────
const SUITS  = ['g', 'e', 's', 'r'];  // Green, Acorns, Bells, Red (matching Rust suit_index)
const VALUES = ['6', '7', '8', '9', 'U', 'O', 'K', '10', 'A'];
const SUIT_SYMBOLS = {
  g: '♣', e: '♦', s: '🔔', r: '♥'   // Unicode proxies; real game would use Bavarian icons
};
const SUIT_LABELS = {g: 'Green', e: 'Acorns', s: 'Bells', r: 'Red'};

function suitClass(suitIdx) {
  return 'suit-' + SUITS[suitIdx];
}
function suitSym(suitIdx) {
  return SUIT_SYMBOLS[SUITS[suitIdx]];
}
function cardLabel(cardIdx) {
  const suit = Math.floor(cardIdx / 9);
  const val  = cardIdx % 9;
  return VALUES[val];
}
function cardSuitIdx(cardIdx) { return Math.floor(cardIdx / 9); }

// Action token → human label
const ACTION_LABELS = {
  40: 'Play card', 41: 'Bid', 42: 'Pass / stop bid', 43: 'Pass cards',
  44: 'Announce trump', 45: 'Ask: pair?', 46: 'Ask: half?',
  47: 'Yes (pair)', 48: 'No (pair)', 49: 'Yes (half)', 50: 'No (half)',
  51: 'Trick won',
};

// ── State ──────────────────────────────────────────────────────────────────────
let ws = null;
let gameState = null;
let aiInfo = null;        // { seat: { logits, probs, entropy } }
let trainStats = null;
let humanSeats = new Set();  // seats controlled by human
let autoPlay = false;
let autoTimer = null;
let lossHistory = [];

// ── WebSocket ──────────────────────────────────────────────────────────────────
function connect() {
  ws = new WebSocket(WS_URL);
  dot(false);

  ws.onopen = () => {
    dot(true);
    log('Connected to server.');
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'game_state') {
      gameState = msg.data;
      aiInfo = msg.ai_info || {};
      renderGame();
    } else if (msg.type === 'train_stats') {
      trainStats = msg.data;
      renderTraining();
    } else if (msg.type === 'error') {
      log('Error: ' + msg.message);
    }
  };

  ws.onclose = () => {
    dot(false);
    log('Disconnected. Retrying in 3s...');
    setTimeout(connect, 3000);
  };
  ws.onerror = () => ws.close();
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function dot(ok) {
  const d = document.getElementById('status-dot');
  d.className = 'dot ' + (ok ? 'dot-ok' : 'dot-error');
}

// ── Rendering ──────────────────────────────────────────────────────────────────

function renderGame() {
  if (!gameState) return;
  const obs = gameState.obs;

  // Scores
  document.getElementById('score-s').textContent = obs.points_my_team ?? 0;
  document.getElementById('score-n').textContent = obs.points_opp_team ?? 0;

  // Trump
  const trumpEl = document.getElementById('trump-suit');
  const trumpLabel = document.getElementById('trump-label');
  if (obs.trump != null) {
    trumpEl.textContent = suitSym(obs.trump);
    trumpEl.className = 'trump-suit ' + suitClass(obs.trump);
    trumpLabel.textContent = SUIT_LABELS[SUITS[obs.trump]];
  } else {
    trumpEl.textContent = '—';
    trumpEl.className = 'trump-suit';
    trumpLabel.textContent = 'No Trump';
  }

  // Hands
  renderHand(0, obs, gameState.human_seats?.includes?.(0) ?? humanSeats.has(0));
  for (let s = 1; s <= 3; s++) {
    renderHand(s, obs, false);
  }

  // Trick
  renderTrick(obs);

  // AI panel
  renderAIPanel(obs);
}

function renderHand(seat, obs, isHuman) {
  const handEl = document.getElementById(`hand-${seat}`);
  handEl.innerHTML = '';

  const count = obs.cards_remaining[seat];

  if (seat === 0 && obs.my_hand_indices) {
    // Show actual cards for POV seat
    const legalCardIdxs = new Set(
      (obs.legal_actions || [])
        .filter(la => la.card_idx != null)
        .map(la => la.card_idx)
    );

    obs.my_hand_indices.forEach(cardIdx => {
      const isLegal = isHuman && legalCardIdxs.has(cardIdx);
      const el = makeCard(cardIdx, isLegal);
      if (isLegal) {
        const la = obs.legal_actions.find(a => a.card_idx === cardIdx);
        el.addEventListener('click', () => {
          send({ cmd: 'human_action', action_list_idx: la.action_list_idx });
          log(`You played: ${cardLabel(cardIdx)} ${suitSym(cardSuitIdx(cardIdx))}`);
        });
      }
      handEl.appendChild(el);
    });
  } else {
    // Show face-down cards for opponents
    for (let i = 0; i < count; i++) {
      const el = document.createElement('div');
      el.className = 'card face-down';
      handEl.appendChild(el);
    }
  }
}

function makeCard(cardIdx, legal) {
  const suit = cardSuitIdx(cardIdx);
  const sym  = suitSym(suit);
  const lbl  = cardLabel(cardIdx);
  const cls  = suitClass(suit);

  const el = document.createElement('div');
  el.className = 'card' + (legal ? ' legal' : '');
  el.title = `${lbl} of ${SUIT_LABELS[SUITS[suit]]}`;
  el.innerHTML = `
    <div class="card-top ${cls}">${lbl}</div>
    <div class="card-center ${cls}">${sym}</div>
    <div class="card-bot ${cls}">${lbl}</div>
  `;
  return el;
}

function renderTrick(obs) {
  const area = document.getElementById('trick-cards');
  const info = document.getElementById('trick-info');
  area.innerHTML = '';

  (obs.current_trick_indices || []).forEach((cardIdx, i) => {
    const el = makeCard(cardIdx, false);
    el.classList.add('in-trick');
    area.appendChild(el);
  });

  const trickNo = obs.trick_number || 1;
  const pos = obs.trick_position || 0;
  info.textContent = `Trick ${trickNo}  ·  Position ${pos + 1}/4`;
}

function renderAIPanel(obs) {
  const container = document.getElementById('ai-seats');
  container.innerHTML = '';

  const roleNames = ['VH (bid winner)', 'MH (partner)', 'LH (left)', 'RH (right)', 'None'];
  const seatNames = ['You (P0)', 'Left (P1)', 'Partner (P2)', 'Right (P3)'];

  for (let seat = 0; seat < 4; seat++) {
    const info = aiInfo[seat] || {};
    const probs = info.probs || [];      // [{label, prob}, ...]
    const entropy = info.entropy ?? 0;
    const isHuman = humanSeats.has(seat);

    const div = document.createElement('div');
    div.className = 'ai-seat-card';

    // Header
    const maxEntropy = Math.log(Math.max(probs.length, 2));
    const entropyPct = Math.min(entropy / maxEntropy * 100, 100);
    const entropyClass = entropyPct < 30 ? 'entropy-low' : entropyPct < 70 ? 'entropy-mid' : 'entropy-high';

    div.innerHTML = `
      <div class="ai-seat-header">
        <div class="ai-seat-name">${seatNames[seat]}${isHuman ? ' 👤' : ''}</div>
        <div class="entropy-bar-wrap">
          <span class="entropy-label">H</span>
          <div class="entropy-bar">
            <div class="entropy-fill ${entropyClass}" style="width:${entropyPct.toFixed(0)}%"></div>
          </div>
        </div>
      </div>
      <div class="policy-rows"></div>
      ${seat !== 0 ? `<button class="btn-takeover${isHuman ? ' active' : ''}" data-seat="${seat}">${isHuman ? '✓ Human' : 'Take over'}</button>` : ''}
    `;

    // Policy table — show top 5 actions
    const rows = div.querySelector('.policy-rows');
    const topProbs = [...probs].sort((a, b) => b.prob - a.prob).slice(0, 5);
    const maxProb = topProbs[0]?.prob ?? 1;

    const table = document.createElement('table');
    table.className = 'policy-table';
    topProbs.forEach((p, i) => {
      const tr = document.createElement('tr');
      if (i === 0) tr.className = 'top-action';
      const barW = Math.round(p.prob / maxProb * 60);
      tr.innerHTML = `
        <td class="action-name">${p.label}</td>
        <td class="prob-bar-cell"><div class="policy-bar" style="width:${barW}px"></div></td>
        <td class="policy-pct">${(p.prob * 100).toFixed(0)}%</td>
      `;
      table.appendChild(tr);
    });
    rows.appendChild(table);

    // Take-over button handler
    div.querySelector('.btn-takeover')?.addEventListener('click', (e) => {
      const s = parseInt(e.target.dataset.seat);
      toggleHuman(s);
    });

    container.appendChild(div);
  }
}

function toggleHuman(seat) {
  if (humanSeats.has(seat)) {
    humanSeats.delete(seat);
    send({ cmd: 'set_seat', seat, human: false });
    log(`Seat ${seat} returned to AI.`);
  } else {
    humanSeats.add(seat);
    send({ cmd: 'set_seat', seat, human: true });
    log(`Seat ${seat} taken over by human.`);
  }
  renderGame();
}

// ── Training panel ─────────────────────────────────────────────────────────────

function renderTraining() {
  if (!trainStats) return;
  document.getElementById('stat-games').textContent = trainStats.game ?? '—';
  const stageLabels = ['Random', 'Heuristic', 'Self-play'];
  document.getElementById('stat-stage').textContent = stageLabels[trainStats.stage] ?? '—';
  const wr = trainStats.win_rate;
  document.getElementById('stat-winrate').textContent = wr != null ? `${(wr * 100).toFixed(0)}%` : '—';
  const loss = trainStats.loss;
  document.getElementById('stat-loss').textContent = loss != null ? loss.toFixed(4) : '—';

  if (trainStats.loss != null) {
    lossHistory.push(trainStats.loss);
    if (lossHistory.length > 100) lossHistory.shift();
    drawLossChart();
  }
}

function drawLossChart() {
  const canvas = document.getElementById('loss-chart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (lossHistory.length < 2) return;

  const W = canvas.width, H = canvas.height;
  const min = Math.min(...lossHistory);
  const max = Math.max(...lossHistory);
  const range = max - min || 1;

  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (lossHistory.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 10) - 5;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under curve
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = 'rgba(88,166,255,0.08)';
  ctx.fill();
}

// ── Log helper ─────────────────────────────────────────────────────────────────
function log(msg) {
  const entries = document.getElementById('log-entries');
  const el = document.createElement('div');
  el.className = 'log-entry';
  const time = new Date().toLocaleTimeString('en', { hour12: false });
  el.innerHTML = `[${time}] ${msg}`;
  entries.prepend(el);
  // Keep max 50 entries
  while (entries.children.length > 50) entries.removeChild(entries.lastChild);
}

// ── Controls ───────────────────────────────────────────────────────────────────
document.getElementById('btn-new-game').addEventListener('click', () => {
  humanSeats.clear();
  send({ cmd: 'new_game' });
  log('New game started.');
});

document.getElementById('btn-proceed').addEventListener('click', () => {
  send({ cmd: 'proceed' });
});

document.getElementById('btn-auto-play').addEventListener('click', () => {
  autoPlay = !autoPlay;
  const btn = document.getElementById('btn-auto-play');
  if (autoPlay) {
    btn.textContent = '⏸ Pause';
    btn.classList.add('btn-accent');
    btn.classList.remove('btn-secondary');
    autoStep();
  } else {
    btn.textContent = '▶ Auto-play';
    btn.classList.remove('btn-accent');
    btn.classList.add('btn-secondary');
    clearTimeout(autoTimer);
  }
});

function autoStep() {
  if (!autoPlay) return;
  send({ cmd: 'proceed' });
  autoTimer = setTimeout(autoStep, 900);
}

document.getElementById('btn-load-ckpt').addEventListener('click', () => {
  const val = document.getElementById('checkpoint-select').value;
  send({ cmd: 'load_checkpoint', checkpoint: val || 'latest' });
  log(`Loading checkpoint: ${val || 'latest'}`);
});

// ── Init ───────────────────────────────────────────────────────────────────────
connect();
// Request initial game state after brief connection delay
setTimeout(() => send({ cmd: 'new_game' }), 800);
