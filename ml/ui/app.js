/**
 * Marjapussi UI â€” app.js
 *
 * Modules (in this file, clearly separated by section headers):
 *   1. Constants & state
 *   2. WebSocket / connection
 *   3. Card DOM helpers
 *   4. Trump display
 *   5. Active player logic
 *   6. Main render orchestrator
 *   7. Hand renderer (selective animation)
 *   8. Trick renderer (selective animation)
 *   9. Bid area / stepper
 *  10. Last-trick strip
 *  11. Event log decoder
 *  12. AI sidebar
 *  13. Debug panel
 *  14. Controls / init
 */

'use strict';

// â”€â”€ 1. Constants & state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const SL = ['Gruen', 'Eichel', 'Schellen', 'Herz'];
const SCOL = ['#15803d', '#78350f', '#b45309', '#dc2626'];
const VL = ['6', '7', '8', '9', 'U', 'O', 'K', '10', 'A'];
const PNAMES = ['Du (P0)', 'P1 Links', 'P2 Partner', 'P3 Rechts'];
// SVG suit files â€” served via /ui static route
const SUIT_IMGS = ['/ui/suits/gruen.svg', '/ui/suits/eichel.svg', '/ui/suits/schell.svg', '/ui/suits/rot.svg'];
const GLYPHS = window.UI_GLYPHS || {};
const EMOJI = GLYPHS.emoji || {};
const SYMBOLS = GLYPHS.symbols || {};

function em(name, fallback = '') {
  return EMOJI[name] || fallback;
}

function sym(name, fallback = '') {
  return SYMBOLS[name] || fallback;
}

const PORT = location.port || 8765;

// Runtime state
let ws = null;
let gs = null;   // current game_state from server
let ainfo = {};     // AI info keyed by seat
let hs = new Set();   // human-controlled seats
let seatViews = {};   // per-seat observations for human-controlled seats
let controllers = {}; // per-seat controller config from server
let checkpointOptions = [];
let pendingSeatCheckpoint = {};
let torchOk = true;
let auto = false;
let autoTm = null;
let viewSeat = 0;
let pendingViewSeat = null;
let followActiveSeat = true;
let preferredManualViewSeat = 0;
const UI_PREFS_KEY = 'marjapussi.ui.prefs.v1';

let debugMode = false;
let allHands = {};  // from debug_state
let dbgTricks = [];
let dbgConfirmed = [];
let dbgPossible = [];
let dbgPredHands = {};
let dbgPredCardProbs = [];
let dbgPredImpossibleMass = {};
let dbgPredHiddenLoss = null;
let dbgInference = {};
const lastKnownHandsBySeat = [[], [], [], []];

// Render-diff state (prevents unnecessary DOM churn)
const prevHand = ['', '', '', ''];  // per seat: sorted card-idx CSV
let prevTrickKey = '';               // CSV of trick card indices
let lastTokenLen = 0;                // last decoded event_tokens length

// Bid stepper and Passing state
let selectedPassCardsBySeat = [new Set(), new Set(), new Set(), new Set()];
let bidStepValue = 120;

// â”€â”€ 2. WebSocket / connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${PORT}/ws`);
  setDot(0);
  ws.onopen = () => setDot(1);
  ws.onclose = () => { setDot(2); setTimeout(connect, 3000); };
  ws.onerror = () => ws.close();
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case 'game_state':
        gs = m.data || null;
        ainfo = m.ai_info || {};
        controllers = gs?.controllers || {};
        seatViews = gs?.seat_views || {};
        if (Number.isInteger(gs?.view_seat)) {
          viewSeat = Math.max(0, Math.min(3, Number(gs.view_seat)));
          if (pendingViewSeat === viewSeat) pendingViewSeat = null;
        }
        checkpointOptions = gs?.checkpoints || checkpointOptions;
        torchOk = gs?.torch_ok !== false;
        hs = new Set(gs?.human_seats || []);
        Object.keys(controllers).forEach(k => {
          const cp = controllers[k]?.checkpoint;
          if (cp != null && pendingSeatCheckpoint[k] === cp) delete pendingSeatCheckpoint[k];
        });
        const active = getActiveSeat(gs?.obs || null);
        if (followActiveSeat && active >= 0) {
          requestViewSeat(active);
        } else if (!followActiveSeat && viewSeat !== preferredManualViewSeat) {
          requestViewSeat(preferredManualViewSeat);
        }
        render();
        if (debugMode) send({ cmd: 'debug_state' });
        break;
      case 'checkpoints':
        checkpointOptions = m.items || [];
        renderAI(gs?.obs || null);
        break;
      case 'debug_state':
        // Server now sends hands, tricks, bitmasks directly
        allHands = m.hands || {};
        dbgTricks = m.tricks || [];
        dbgConfirmed = m.confirmed_bitmasks || [];
        dbgPossible = m.possible_bitmasks || [];
        dbgPredHands = m.predicted_hands || {};
        dbgPredCardProbs = m.predicted_card_probs || [];
        dbgPredImpossibleMass = m.predicted_impossible_mass || {};
        dbgPredHiddenLoss = m.predicted_hidden_loss || null;
        dbgInference = m.inference_stats || {};
        if (m.seat_policy) ainfo = m.seat_policy;
        renderDebugPanel(gs?.obs);
        render(); // update table face-up cards
        break;
      case 'error':
        evLog(em('cross', 'x'), m.message, 'err-ev');
        break;
    }
  };
}

function send(o) { if (ws?.readyState === 1) ws.send(JSON.stringify(o)); }

function setDot(s) {
  document.getElementById('dot').className = 'dot ' + (['', 'ok', 'err'][s] || '');
}

// â”€â”€ 3. Card DOM helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/** Create an <img> element for a suit symbol. Always created via DOM (not innerHTML). */
function mkSuitImg(suitIdx, sizePx) {
  const img = document.createElement('img');
  img.src = SUIT_IMGS[suitIdx] ?? SUIT_IMGS[0];
  img.width = img.height = sizePx;
  img.style.cssText = 'display:block;object-fit:contain;pointer-events:none;flex-shrink:0';
  img.draggable = false;
  return img;
}

/**
 * Build a face-up card element.
 * @param {number} idx     Card index 0-35 (suit*9 + val)
 * @param {string} cls     Extra CSS classes (e.g. 'legal', 'trick')
 * @param {boolean} isNew  If true, adds .card-new animation class
 */
function mkCard(idx, cls = '', isNew = false) {
  const su = (idx / 9) | 0;
  const va = idx % 9;
  const vl = VL[va];
  const c = SCOL[su];

  const el = document.createElement('div');
  el.className = 'card' + (cls ? ' ' + cls : '') + (isNew ? ' card-new' : '');
  el.title = vl + ' ' + SL[su];
  el.dataset.ci = idx;  // used for diff-checking

  // Top corner: value + small suit
  const top = document.createElement('div');
  top.style.cssText = 'display:flex;flex-direction:column;align-items:flex-start';
  const tv = document.createElement('div');
  tv.className = 'cv'; tv.style.color = c; tv.textContent = vl;
  top.appendChild(tv);
  top.appendChild(mkSuitImg(su, 12));

  // Center: large suit symbol
  const ctr = document.createElement('div');
  ctr.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)';
  ctr.appendChild(mkSuitImg(su, 28));

  // Bottom corner (rotated)
  const bot = document.createElement('div');
  bot.className = 'cbot';
  const bv = document.createElement('div');
  bv.className = 'cv'; bv.style.color = c; bv.textContent = vl;
  bot.appendChild(bv);
  bot.appendChild(mkSuitImg(su, 12));

  el.appendChild(top); el.appendChild(ctr); el.appendChild(bot);
  return el;
}

/** Face-down card. */
function mkFD() {
  const e = document.createElement('div');
  e.className = 'card fd';
  return e;
}

// â”€â”€ 4. Trump display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

let _lastTrump = undefined;
function setTrump(ti) {
  if (_lastTrump === ti) return;  // skip if unchanged
  _lastTrump = ti;
  const trumpSym = document.getElementById('trump-sym');
  trumpSym.innerHTML = '';
  if (ti != null) {
    trumpSym.appendChild(mkSuitImg(ti, 30));
  } else {
    trumpSym.textContent = sym('em_dash', '-');
  }
  document.getElementById('trump-lbl').textContent = ti != null ? SL[ti] : 'Kein Trumpf';
}

// â”€â”€ 5. Active player logic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/**
 * Returns the seat (0-3) that should act next, or -1 if unknown.
 * Server always sends obs from P0's POV.
 * If legal_actions is non-empty â†’ it is P0's turn (server only sends actions for the acting player).
 * During a trick: next = (leader + cards_played) % 4.
 */
function getActiveSeat(obs) {
  if (!obs) return -1;
  return obs.active_player ?? -1;
}

function ctrlForSeat(seat) {
  return controllers?.[String(seat)] || controllers?.[seat] || null;
}

function effectiveMode(seat) {
  const ctrl = ctrlForSeat(seat);
  if (!ctrl) return hs.has(seat) ? 'human' : 'heuristic';
  return ctrl.effective_mode || ctrl.mode || 'heuristic';
}

function seatObs(seat) {
  if (seat === 0) return gs?.obs || null;
  return seatViews?.[String(seat)] || seatViews?.[seat] || null;
}

function actionObsForSeat(activeSeat) {
  if (activeSeat === 0) return gs?.obs || null;
  // In debug mode we allow manual action selection for non-human seats too.
  if (effectiveMode(activeSeat) === 'human' || debugMode) return seatObs(activeSeat);
  return null;
}

function isSeatHumanControlled(seat) {
  return effectiveMode(seat) === 'human';
}

function normalizeSeat(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(3, Math.trunc(n)));
}

function requestViewSeat(seat) {
  const target = normalizeSeat(seat);
  if (target === viewSeat || pendingViewSeat === target) return;
  pendingViewSeat = target;
  send({ cmd: 'set_view_seat', seat: target });
}

function currentViewSeat(activeSeat) {
  if (followActiveSeat && activeSeat >= 0) return activeSeat;
  return viewSeat;
}

function updateTurnState(obs, activeSeat, tableViewSeat) {
  const el = document.getElementById('turn-state');
  if (!el) return;
  if (!obs || activeSeat < 0) {
    el.textContent = 'Kein Spiel';
    return;
  }
  const mode = effectiveMode(activeSeat);
  const canAct = mode === 'human' || debugMode;
  const activeLabel = PNAMES[activeSeat] || `P${activeSeat}`;
  const viewLabel = PNAMES[tableViewSeat] || `P${tableViewSeat}`;
  el.textContent = `Aktiv: ${activeLabel} (${mode}) | View: ${viewLabel} | Input: ${canAct ? 'frei' : 'gesperrt'}`;
}

const LAYOUT_STORAGE_KEY = 'marjapussi.ui.layout.v1';
const LAYOUT_DEFAULTS = { side: 260, col3: 220 };
const LAYOUT_LIMITS = {
  boardMin: 420,
  sideMin: 180,
  sideMax: 560,
  col3Min: 160,
  col3Max: 520,
};

function clamp(v, lo, hi) {
  if (hi < lo) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function readLayoutWidths() {
  const cs = getComputedStyle(document.documentElement);
  const side = Number.parseFloat(cs.getPropertyValue('--side-w')) || LAYOUT_DEFAULTS.side;
  const col3 = Number.parseFloat(cs.getPropertyValue('--col3-w')) || LAYOUT_DEFAULTS.col3;
  return { side, col3 };
}

function writeLayoutWidths(side, col3, persist = true) {
  const rootStyle = document.documentElement.style;
  if (Number.isFinite(side)) rootStyle.setProperty('--side-w', `${Math.round(side)}px`);
  if (Number.isFinite(col3)) rootStyle.setProperty('--col3-w', `${Math.round(col3)}px`);
  if (persist) {
    try { localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify({ side, col3 })); } catch (_) { }
  }
}

function clampLayoutForViewport(side, col3) {
  const main = document.getElementById('main');
  const col3El = document.getElementById('col3');
  if (!main || !col3El) return { side, col3 };

  const total = main.clientWidth || 0;
  const col3Visible = getComputedStyle(col3El).display !== 'none';
  if (window.innerWidth <= 1120) return { side, col3 };

  if (!col3Visible) {
    const sideMax = Math.max(LAYOUT_LIMITS.sideMin, total - LAYOUT_LIMITS.boardMin);
    return {
      side: clamp(side, LAYOUT_LIMITS.sideMin, Math.min(sideMax, LAYOUT_LIMITS.sideMax)),
      col3,
    };
  }

  const available = Math.max(0, total - LAYOUT_LIMITS.boardMin);
  const sideMax = Math.max(LAYOUT_LIMITS.sideMin, Math.min(LAYOUT_LIMITS.sideMax, available - LAYOUT_LIMITS.col3Min));
  const nextSide = clamp(side, LAYOUT_LIMITS.sideMin, sideMax);
  const col3Max = Math.max(LAYOUT_LIMITS.col3Min, Math.min(LAYOUT_LIMITS.col3Max, available - nextSide));
  const nextCol3 = clamp(col3, LAYOUT_LIMITS.col3Min, col3Max);
  return { side: nextSide, col3: nextCol3 };
}

function initColumnResizers() {
  const main = document.getElementById('main');
  const handleMain = document.getElementById('handle-main');
  const handleSide = document.getElementById('handle-side');
  const col3El = document.getElementById('col3');
  if (!main || !handleMain || !handleSide || !col3El) return;

  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Number.isFinite(parsed?.side) && Number.isFinite(parsed?.col3)) {
        const clamped = clampLayoutForViewport(parsed.side, parsed.col3);
        writeLayoutWidths(clamped.side, clamped.col3, false);
      }
    }
  } catch (_) { }

  const applyViewportClamp = () => {
    const w = readLayoutWidths();
    const clamped = clampLayoutForViewport(w.side, w.col3);
    writeLayoutWidths(clamped.side, clamped.col3, false);
  };
  applyViewportClamp();
  window.addEventListener('resize', applyViewportClamp);

  let dragKind = null; // 'main' | 'side'

  const onMove = ev => {
    if (!dragKind) return;
    if (window.innerWidth <= 1120) return;
    const rect = main.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const total = rect.width;
    const col3Visible = getComputedStyle(col3El).display !== 'none';
    const w = readLayoutWidths();

    if (dragKind === 'main') {
      if (!col3Visible) {
        const rawSide = total - x;
        const sideMax = Math.max(LAYOUT_LIMITS.sideMin, total - LAYOUT_LIMITS.boardMin);
        writeLayoutWidths(clamp(rawSide, LAYOUT_LIMITS.sideMin, Math.min(sideMax, LAYOUT_LIMITS.sideMax)), w.col3, false);
        return;
      }
      const rawSide = total - x - w.col3;
      const available = Math.max(0, total - LAYOUT_LIMITS.boardMin);
      const sideMax = Math.max(LAYOUT_LIMITS.sideMin, Math.min(LAYOUT_LIMITS.sideMax, available - LAYOUT_LIMITS.col3Min));
      writeLayoutWidths(clamp(rawSide, LAYOUT_LIMITS.sideMin, sideMax), w.col3, false);
      return;
    }

    if (dragKind === 'side' && col3Visible) {
      const rawCol3 = total - x;
      const available = Math.max(0, total - LAYOUT_LIMITS.boardMin);
      const col3Max = Math.max(LAYOUT_LIMITS.col3Min, Math.min(LAYOUT_LIMITS.col3Max, available - w.side));
      writeLayoutWidths(w.side, clamp(rawCol3, LAYOUT_LIMITS.col3Min, col3Max), false);
    }
  };

  const stopDrag = () => {
    if (!dragKind) return;
    dragKind = null;
    document.body.classList.remove('col-resizing');
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', stopDrag);
    window.removeEventListener('pointercancel', stopDrag);
    const w = readLayoutWidths();
    const clamped = clampLayoutForViewport(w.side, w.col3);
    writeLayoutWidths(clamped.side, clamped.col3);
  };

  const startDrag = kind => ev => {
    if (ev.button !== 0) return;
    dragKind = kind;
    document.body.classList.add('col-resizing');
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', stopDrag);
    window.addEventListener('pointercancel', stopDrag);
    ev.preventDefault();
  };

  handleMain.addEventListener('pointerdown', startDrag('main'));
  handleSide.addEventListener('pointerdown', startDrag('side'));
}

// â”€â”€ 6. Main render orchestrator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function render() {
  if (!gs) return;
  const obs = gs.obs;
  const noGame = document.getElementById('no-game');
  if (!obs) { noGame.style.display = ''; return; }
  noGame.style.display = 'none';

  // Scores
  document.getElementById('spts').textContent = obs.points_my_team ?? 0;
  document.getElementById('opts').textContent = obs.points_opp_team ?? 0;

  // Trump
  setTrump(obs.trump);

  const active = getActiveSeat(obs);
  const tableViewSeat = currentViewSeat(active);
  if (followActiveSeat && active >= 0) requestViewSeat(active);

  // Hands (selective re-render)
  for (let s = 0; s < 4; s++) renderHand(s, obs, tableViewSeat);
  renderTrick(obs);

  // Active + view ring
  for (let s = 0; s < 4; s++) {
    const seatEl = document.getElementById('p' + s);
    seatEl?.classList.toggle('player-active', s === active);
    seatEl?.classList.toggle('player-view', !debugMode && s === tableViewSeat);
  }
  updateTurnState(obs, active, tableViewSeat);
  const viewSel = document.getElementById('view-seat');
  if (viewSel) {
    const desired = (followActiveSeat && active >= 0) ? 'active' : String(viewSeat);
    if (viewSel.value !== desired) viewSel.value = desired;
  }

  // Bid/action area (shows active player's non-card actions if human or debug)
  renderBidArea(obs, active);

  // Event log (only if token stream grew)
  renderEventLog(obs);

  // AI sidebar
  renderAI(obs);

  // Last-trick strip (from end-of-game outcome, if available)
  const info = gs.info;
  if (info?.tricks?.length) {
    const lt = info.tricks[info.tricks.length - 1];
    showLastTrick(lt.cards, lt.winner, lt.points);
  }
}

// â”€â”€ 7. Hand renderer (selective animation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function renderHand(s, obs, tableViewSeat) {
  const el = document.getElementById('h' + s);
  if (!el) return;

  const active = getActiveSeat(obs);
  const isMyTurn = active === s;
  const isHuman = effectiveMode(s) === 'human';
  const activeSeatObs = isMyTurn ? actionObsForSeat(active) : null;
  // Never fall back to P0 observation when a non-P0 human seat is active.
  const activeObs = isMyTurn ? (activeSeatObs || (active === 0 ? obs : null)) : null;
  const canAct = isMyTurn && (isHuman || debugMode) && !!activeObs;
  const phaseObs = activeObs || obs;
  const isPassing = (phaseObs.phase === "PassingForth" || phaseObs.phase === "PassingBack");
  const hasSequentialPass = isPassing && !!activeObs && (activeObs.legal_actions || []).some(
    la => la.action_token === 52 && la.card_idx != null
  );
  const directPassCombos = (isPassing && !!activeObs && !hasSequentialPass)
    ? (activeObs.legal_actions || []).filter(
      la => Array.isArray(la.pass_cards) && la.pass_cards.length === 4
    )
    : [];
  const passCandidates = new Set();
  directPassCombos.forEach(la => (la.pass_cards || []).forEach(ci => passCandidates.add(ci)));
  const selectedPassCards = selectedPassCardsBySeat[s];

  if (canAct) {
    if (isPassing) {
      if (!activeObs) {
        selectedPassCards.clear();
      } else if (hasSequentialPass) {
        selectedPassCards.clear();
        (phaseObs.pass_selection_indices || []).forEach(ci => selectedPassCards.add(ci));
      } else if (selectedPassCards.size) {
        // Keep manual/direct selection coherent with currently legal pass candidates.
        [...selectedPassCards].forEach(ci => {
          if (!passCandidates.has(ci)) selectedPassCards.delete(ci);
        });
      }
    } else {
      selectedPassCards.clear();
    }
  }

  // Legal actions mapping (only relevant for the active player)
  const lm = new Map();
  if (isMyTurn && activeObs) {
    (activeObs.legal_actions || []).forEach(la => {
      if (la.card_idx != null) lm.set(la.card_idx, la.action_list_idx);
    });
  }

  // Hand visibility policy:
  // - Debug mode: full information (all seats face-up via debug payload where available).
  // - Normal mode:
  //   1) always show selected view seat,
  //   2) keep all human-controlled seats visible for playability/continuity.
  const debugHand = allHands[s] || allHands[String(s)] || null;
  const seatView = seatObs(s);
  const isHumanSeat = isSeatHumanControlled(s);
  const canShowViewSeat = s === tableViewSeat;
  let faceUpCards = null;
  if (debugMode) {
    faceUpCards = (s === 0) ? obs.my_hand_indices : debugHand;
  } else if (isHumanSeat || canShowViewSeat) {
    faceUpCards = (s === 0) ? obs.my_hand_indices : seatView?.my_hand_indices;
  }

  // Persist last known cards so hands do not disappear while waiting for a fresh POV view.
  if (Array.isArray(faceUpCards) && faceUpCards.length > 0) {
    lastKnownHandsBySeat[s] = [...faceUpCards];
  } else if (!debugMode && (isHumanSeat || canShowViewSeat) && lastKnownHandsBySeat[s].length > 0) {
    faceUpCards = [...lastKnownHandsBySeat[s]];
  }

  if (faceUpCards?.length) {
    const newKey = [...faceUpCards].sort((a, b) => a - b).join(',');

    // To ensure cards always have the correct click listener mapping (especially when receiving 4 cards mid-phase), 
    // we must aggressively bust the cache whenever the exact array of cards changes, 
    // OR if we entered a passing phase.
    const cacheKey = newKey + '|' + isPassing + '|' + active;

    if (cacheKey !== prevHand[s]) {
      const prev = new Set(prevHand[s] ? prevHand[s].split('|')[0].split(',').map(Number) : []);
      const confirmed = obs.confirmed_bitmasks?.[s > 0 ? s - 1 : 0] || [];
      el.innerHTML = '';

      const sortedHand = [...faceUpCards].sort((a, b) => {
        const suA = Math.floor(a / 9), vaA = a % 9;
        const suB = Math.floor(b / 9), vaB = b % 9;
        if (suA !== suB) return suB - suA;
        return vaB - vaA;
      });

      sortedHand.forEach(i => {
        const ai = lm.get(i);
        const passLegal = isPassing && (
          hasSequentialPass
            ? (ai !== undefined || selectedPassCards.has(i))
            : passCandidates.has(i)
        );
        const legalPlay = canAct && (
          ai !== undefined || passLegal
        );
        let cls = legalPlay ? 'legal' : '';

        if (s > 0 && debugMode) {
          cls += confirmed[i] ? ' debug-confirm' : ' debug-certain';
        }
        if (isPassing && selectedPassCards.has(i)) cls += ' selected';

        const card = mkCard(i, cls.trim(), !prev.has(i));

        // Always attach the listener so cached DOM elements still react!
        card.addEventListener('click', () => {
          // Re-evaluate current state dynamically (since DOM nodes persist)
          const activeNow = getActiveSeat(gs?.obs);
          const currSeatObs = activeNow >= 0 ? actionObsForSeat(activeNow) : null;
          const currActionObs = activeNow >= 0 ? (currSeatObs || (activeNow === 0 ? gs?.obs : null)) : null;
          if (!currActionObs) return;
          const currLm = new Map();
          (currActionObs?.legal_actions || []).forEach(la => {
            if (la.card_idx != null) currLm.set(la.card_idx, la.action_list_idx);
          });
          const currAi = currLm.get(i);
          const currPassing = (currActionObs?.phase === "PassingForth" || currActionObs?.phase === "PassingBack");
          const currSeqPass = currPassing && (currActionObs?.legal_actions || []).some(
            la => la.action_token === 52 && la.card_idx != null
          );
          const currPassCandidates = new Set();
          if (currPassing && !currSeqPass) {
            (currActionObs?.legal_actions || []).forEach(la => {
              if (Array.isArray(la.pass_cards) && la.pass_cards.length === 4) {
                la.pass_cards.forEach(ci => currPassCandidates.add(ci));
              }
            });
          }
          const canActNow = (activeNow === s) && (effectiveMode(s) === 'human' || debugMode);
          const passLegalNow = currPassing && (
            currSeqPass
              ? (currAi !== undefined || selectedPassCards.has(i))
              : currPassCandidates.has(i)
          );
          const legalNow = canActNow && (
            currAi !== undefined || passLegalNow
          );
          if (!legalNow) return; // ignore clicks if currently not legal

          if (currPassing) {
            if (currSeqPass) {
              if (currAi !== undefined) {
                send({ cmd: 'human_action', action_list_idx: currAi });
              }
            } else {
              if (selectedPassCards.has(i)) selectedPassCards.delete(i);
              else if (selectedPassCards.size < 4) selectedPassCards.add(i);
              if (selectedPassCards.size === 4) {
                const chosen = [...selectedPassCards].sort((a, b) => a - b);
                const match = (currActionObs?.legal_actions || []).find(la => {
                  const pc = (la.pass_cards || []).slice().sort((a, b) => a - b);
                  return pc.length === 4 && pc.every((v, idx) => v === chosen[idx]);
                });
                if (match) {
                  send({ cmd: 'human_action', action_list_idx: match.action_list_idx });
                  selectedPassCards.clear();
                } else {
                  evLog(em('cross', 'x'), 'Diese 4 Karten sind so nicht legal schiebbar.', 'err-ev');
                }
              }
              render();
            }
          } else if (currAi !== undefined) {
            send({ cmd: 'human_action', action_list_idx: currAi });
            evLog(em('play', 'play'), `P${s} spielt: <b>${VL[i % 9]} ${SL[(i / 9) | 0]}</b>`);
          }
        });

        // Set initial visual accessibility
        if (!legalPlay) {
          card.style.opacity = '0.5';
          if (!canAct) card.style.cursor = 'default';
          else card.style.cursor = 'not-allowed';
        }
        el.appendChild(card);
      });
      prevHand[s] = cacheKey;
    } else {
      el.querySelectorAll('.card').forEach(card => {
        const ci = +card.dataset.ci;
        const ai = lm.get(ci);
        const passLegal = isPassing && (
          hasSequentialPass
            ? (ai !== undefined || selectedPassCards.has(ci))
            : passCandidates.has(ci)
        );
        const legalPlay = canAct && (
          ai !== undefined || passLegal
        );

        card.classList.toggle('legal', legalPlay);
        if (isPassing) card.classList.toggle('selected', selectedPassCards.has(ci));
        else card.classList.remove('selected');

        if (!legalPlay) {
          card.style.opacity = '0.5';
          card.style.cursor = canAct ? 'not-allowed' : 'default';
        } else {
          card.style.opacity = '1';
          card.style.cursor = 'pointer';
        }
      });
    }

  } else {
    // â€”â€” Face-down cards â€”â€”
    const n = obs.cards_remaining?.[s] ?? 0;
    if (el.children.length !== n || el.querySelector('.card:not(.fd)')) {
      el.innerHTML = '';
      for (let i = 0; i < n; i++) el.appendChild(mkFD());
      prevHand[s] = '';
    }
  }
}

// â€”â€” 8. Trick renderer (animate only newest card) â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”

function renderTrick(obs) {
  let trickIdxs = obs.current_trick_indices || [];
  let trickPlayers = obs.current_trick_players || [];
  let isLingering = false;

  // The server resolves tricks instantly, so we never see trickIdxs.length === 4.
  // When a trick is empty but we're past trick 1, scan event_tokens backwards
  // to find the last 4 played cards so we can linger them.
  if (trickIdxs.length === 0 && (obs.trick_number || 1) > 1) {
    const tokens = obs.event_tokens || [];
    const recentCards = [];
    const recentPlayers = [];
    // Only linger if the very last card-related event was the end of a trick
    // (Meaning we haven't started playing cards for the *current* trick yet)
    for (let i = tokens.length - 1; i >= 2; i--) {
      // 51 = Stich gewonnen (Trick won). If we see this, we know the trick just ended.
      if (tokens[i] === 51) continue;
      // Tokens 47-50 are Answer Pair/Half. Skip them.
      if (tokens[i] >= 47 && tokens[i] <= 50) continue;
      // 45, 46 = Ask Pair/Half. Skip.
      if (tokens[i] === 45 || tokens[i] === 46) {
        // These are followed by the player and then the suit token
        i -= 2;
        continue;
      }

      // Card play pattern in Rust: [player(20-23), ACT_PLAY(40), card(70-105)]
      if (tokens[i] >= 70 && tokens[i] <= 105 && tokens[i - 1] === 40 && tokens[i - 2] >= 20 && tokens[i - 2] <= 23) {
        // We found a Card Played event
        recentCards.unshift(tokens[i] - 70);
        recentPlayers.unshift(tokens[i - 2] - 20);
        i -= 2; // skip the rest of this triplet
        if (recentCards.length === 4) break;
      }
    }
    if (recentCards.length === 4) {
      trickIdxs = recentCards;
      trickPlayers = recentPlayers;
      isLingering = true;
    }
  }

  const newKey = trickIdxs.join(',') + (isLingering ? '-L' : '');
  if (newKey === prevTrickKey) return;  // no change

  const tc = document.getElementById('trick');
  const prevSet = new Set(prevTrickKey ? prevTrickKey.replace('-L', '').split(',').map(Number) : []);
  tc.innerHTML = '';
  trickIdxs.forEach((ci, i) => {
    const isNew = !isLingering && !prevSet.has(ci);
    const card = mkCard(ci, 'trick' + (isLingering ? ' lingering' : ''), isNew);
    if (!isLingering) card.title += ' ' + sym('right_arrow', '->') + ' ' + PNAMES[trickPlayers[i] ?? i];
    tc.appendChild(card);
  });
  prevTrickKey = newKey;

  const tinfo = document.getElementById('trick-info');
  if (isLingering) {
    tinfo.textContent = `Stich ${obs.trick_number - 1} abgeschlossen`;
  } else {
    tinfo.textContent = `Stich ${obs.trick_number ?? 1} ${sym('middle_dot', '-')} ${trickIdxs.length}/4 Karten`;
  }
}
// â”€â”€ 9. Bid area / stepper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function renderBidArea(obs, active) {
  const ba = document.getElementById('bid-area');
  if (!obs || active < 0) { ba.innerHTML = ''; return; }

  const isHuman = effectiveMode(active) === 'human';
  const canAct = isHuman || debugMode;
  if (!canAct) {
    ba.innerHTML = '';
    document.getElementById('p0').appendChild(ba);
    return;
  }

  // Move bid area to active player so buttons appear under their cards
  document.getElementById('p' + active).appendChild(ba);

  const seatObsForActive = actionObsForSeat(active);
  const actObs = seatObsForActive || (active === 0 ? obs : null);
  if (!actObs) {
    ba.dataset.ahash = '';
    ba.innerHTML = '';
    const wait = document.createElement('div');
    wait.className = 'bid-btn';
    wait.style.opacity = '0.6';
    wait.style.cursor = 'default';
    wait.textContent = 'Warte auf legale Aktionen...';
    ba.appendChild(wait);
    return;
  }
  const legal = actObs.legal_actions || [];
  const selectedPassCards = selectedPassCardsBySeat[active];

  // Passing Cards (Gib ab)
  const isPassing = (actObs.phase === "PassingForth" || actObs.phase === "PassingBack");
  if (canAct && isPassing) {
    ba.dataset.ahash = ''; // force re-render on selection changes
    ba.innerHTML = '';
    const hasSequentialPass = legal.some(la => la.action_token === 52 && la.card_idx != null);
    if (hasSequentialPass) {
      const selected = actObs.pass_selection_indices || [];
      const target = actObs.pass_selection_target || 4;
      const txt = document.createElement('div');
      txt.className = 'bid-btn act';
      txt.style.opacity = '1';
      txt.style.cursor = 'default';
      if (selected.length < target) {
        txt.textContent = `Waehle Karten (${selected.length}/${target})`;
      } else {
        txt.textContent = `Schieben...`;
      }
      ba.appendChild(txt);
    } else {
      const directPassCombos = legal.filter(
        la => Array.isArray(la.pass_cards) && la.pass_cards.length === 4
      );
      const btn = document.createElement('button');
      btn.className = 'bid-btn act';
      if (selectedPassCards.size === 4) {
        const selArr = [...selectedPassCards].sort((a, b) => a - b);
        const match = directPassCombos.find(la => {
          const pc = (la.pass_cards || []).slice().sort((a, b) => a - b);
          return pc.length === 4 && pc.every((v, idx) => v === selArr[idx]);
        });
        btn.textContent = match ? `Schieben (4/4)` : `Auswahl nicht legal (4/4)`;
        btn.onclick = () => {
          if (match) send({ cmd: 'human_action', action_list_idx: match.action_list_idx });
          selectedPassCards.clear();
        };
        if (!match) {
          btn.disabled = true;
          btn.style.opacity = '0.5';
        }
      } else {
        btn.textContent = `Waehle 4 Karten (${selectedPassCards.size}/4)`;
        btn.disabled = true;
        btn.style.opacity = '0.5';
      }
      ba.appendChild(btn);
    }
    return;
  }

  // Standard non-card actions
  const nonCard = legal.filter(la => la.card_idx == null && la.pass_cards == null);
  if (!canAct || !nonCard.length) { ba.innerHTML = ''; return; }

  // Avoid re-rendering if same action set (hash)
  const aHash = [
    String(active),
    ...nonCard.map(
      la => `${la.action_list_idx}_${la.action_token}_${la.bid_value ?? ''}_${la.suit_idx ?? ''}`
    )
  ].join('|');
  if (ba.dataset.ahash === aHash) return;
  ba.dataset.ahash = aHash;
  ba.innerHTML = '';

  const bidActions = nonCard.filter(la => la.action_token === 41);
  const otherActions = nonCard.filter(la => la.action_token !== 41 && la.action_token !== 43);

  // â”€â”€ Bid stepper â”€â”€
  if (bidActions.length > 0) {
    const bidVals = bidActions.map(la => la.bid_value).filter(v => v != null).sort((a, b) => a - b);
    const minB = bidVals[0], maxB = bidVals[bidVals.length - 1];

    // Clamp and snap bidStepValue to nearest legal value
    bidStepValue = Math.max(minB, Math.min(maxB, bidStepValue));
    if (!bidVals.includes(bidStepValue)) {
      bidStepValue = bidVals.reduce((p, c) =>
        Math.abs(c - bidStepValue) < Math.abs(p - bidStepValue) ? c : p);
    }

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:4px';

    const valDisp = document.createElement('span');
    valDisp.style.cssText = 'font:700 18px monospace;color:var(--warn);min-width:44px;text-align:center;padding:2px 6px;border:1px solid rgba(210,153,34,.35);border-radius:6px';
    valDisp.textContent = bidStepValue;

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'bid-btn act';
    confirmBtn.style.marginLeft = '8px';
    confirmBtn.textContent = `Biete ${bidStepValue}`;

    const step = delta => {
      const idx = bidVals.indexOf(bidStepValue);
      const ni = Math.max(0, Math.min(bidVals.length - 1, idx + delta));
      bidStepValue = bidVals[ni];
      valDisp.textContent = bidStepValue;
      confirmBtn.textContent = `Biete ${bidStepValue}`;
    };

    const mkStep = (lbl, d) => {
      const b = document.createElement('button');
      b.className = 'bid-btn'; b.textContent = lbl; b.style.minWidth = '38px';
      b.addEventListener('click', () => step(d));
      return b;
    };

    const step50 = Math.max(1, bidVals.findIndex(v => v >= bidVals[0] + 50));
    row.appendChild(mkStep(`${sym('minus_sign', '-')}50`, -step50));
    row.appendChild(mkStep(`${sym('minus_sign', '-')}5`, -1));
    row.appendChild(valDisp);
    row.appendChild(mkStep('+5', +1));
    row.appendChild(mkStep('+50', +step50));
    row.appendChild(confirmBtn);

    confirmBtn.addEventListener('click', () => {
      const la = bidActions.find(a => a.bid_value === bidStepValue);
      if (la) {
        send({ cmd: 'human_action', action_list_idx: la.action_list_idx });
        evLog(em('money', '$'), `Du bietest: <b>${bidStepValue}</b>`, 'bid-ev');
      }
    });
    ba.appendChild(row);
  }

  // â”€â”€ Other non-bid actions (Passe, Trumpf, Paar?, etc.) â”€â”€
  otherActions.forEach(la => {
    const btn = document.createElement('button');
    let label = '', cls = 'bid-btn';
    // Build fully-qualified label so every action looks unique
    switch (la.action_token) {
      case 42: label = 'Passe'; cls += ' pass'; break;
      case 43: label = 'Gib ab'; cls += ' pass'; break;
      case 44:
        label = `${em('trophy', 'T')} Trumpf` + (la.suit_idx != null ? `: ${SL[la.suit_idx]}` : '');
        cls += ' act'; break;
      case 45: label = 'Paar?'; cls += ' act'; break;
      case 46:
        label = 'Halb?' + (la.suit_idx != null ? ` ${SL[la.suit_idx]}` : '');
        cls += ' act'; break;
      case 47:
        label = `${em('check', 'Y')} Ja (Paar)` + (la.suit_idx != null ? ` ${SL[la.suit_idx]}` : '');
        cls += ' act'; break;
      case 48:
        label = `${em('cross', 'N')} Nein (Paar)` + (la.suit_idx != null ? ` ${SL[la.suit_idx]}` : '');
        cls += ' pass'; break;
      case 49:
        label = `${em('check', 'Y')} Ja (Halb)` + (la.suit_idx != null ? ` ${SL[la.suit_idx]}` : '');
        cls += ' act'; break;
      case 50:
        label = `${em('cross', 'N')} Nein (Halb)` + (la.suit_idx != null ? ` ${SL[la.suit_idx]}` : '');
        cls += ' pass'; break;
      default: label = `Aktion ${la.action_token}`;
    }
    btn.className = cls; btn.textContent = label;
    btn.addEventListener('click', () => {
      send({ cmd: 'human_action', action_list_idx: la.action_list_idx });
      evLog(em('play', 'act'), `Du: <b>${label}</b>`, 'bid-ev');
    });
    ba.appendChild(btn);
  });
}

// â”€â”€ 10. Last-trick strip â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function showLastTrick(cards, winner, pts) {
  const row = document.getElementById('last-trick-row');
  const ltc = document.getElementById('lt-cards');
  if (!cards?.length) { row.style.display = 'none'; return; }
  const key = cards.join('|') + '|' + winner;
  if (row.dataset.key === key) return;  // no change
  row.dataset.key = key;
  row.style.display = '';
  document.getElementById('lt-label').textContent =
    `Letzter Stich ${sym('middle_dot', '-')} ${PNAMES[winner] ?? 'P' + winner} gewinnt (${pts} Pkt.)`;
  ltc.innerHTML = '';
  cards.forEach(cardStr => {
    const idx = parseCardStr(cardStr);
    if (idx >= 0) ltc.appendChild(mkCard(idx, '', false));
    else {
      const sp = document.createElement('span');
      sp.style.cssText = 'font-size:10px;color:var(--dim);padding:2px';
      sp.textContent = cardStr;
      ltc.appendChild(sp);
    }
  });
}

/** Parse Rust Display format: "6 Gruen", "A Herz", etc. */
function parseCardStr(s) {
  const VMAP = { '6': 0, '7': 1, '8': 2, '9': 3, 'U': 4, 'O': 5, 'K': 6, '10': 7, 'A': 8 };
  const p = s.trim().split(/\s+/);
  if (p.length < 2) return -1;
  const vi = VMAP[p[0]];
  const suitRaw = p[1].toLowerCase();
  const suit = suitRaw === 'gr\u00fcn' ? 'gruen' : suitRaw;
  const SMAP = {
    gruen: 0, eichel: 1, schellen: 2, herz: 3,
    green: 0, acorns: 1, bells: 2, red: 3
  };
  const si = SMAP[suit];
  return (vi == null || si == null) ? -1 : si * 9 + vi;
}

// â”€â”€ 11. Event log decoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/** Decode event_tokens into human-readable log entries.
 *  Token constants match observation.rs:
 *    1       = START_GAME (followed by role token 30-34)
 *    10-18   = Stich separator (10+n for trick n)
 *    20-23   = Player token (20+seat)
 *    30-34   = Role tokens (VH/MH/LH/RH/none)
 *    40      = Play card
 *    41      = Bid
 *    42      = Pass
 *    43      = Pass cards
 *    44      = Trump
 *    45      = Ask pair?
 *    46      = Ask half?
 *    47      = Yes (pair) [+ suit]
 *    48      = No (pair)
 *    49      = Yes (half) [+ suit]
 *    50      = No (half) [+ suit]
 *    51      = Trick won (followed by player token)
 *    60-63   = Suit token (60+suit)
 *    70-105  = Card token (70+card_idx)
 *    110     = Unknown/hidden card
 *    120+    = Bid value (120 + (val-120)/5)
 */
function renderEventLog(obs) {
  if (!obs) return;
  const tokens = obs.event_tokens || [];
  if (tokens.length === lastTokenLen) return;
  lastTokenLen = tokens.length;

  const el = document.getElementById('evlog');
  el.innerHTML = '';

  const raw = [];
  let i = 0;
  const peek = () => tokens[i];
  const eat = () => tokens[i++];
  const isP = t => t >= 20 && t <= 23;
  const isSu = t => t >= 60 && t <= 63;
  const isCa = t => t >= 70 && t <= 105;
  const isBV = t => t >= 120;
  const ROLES = { 30: 'VH', 31: 'MH', 32: 'LH', 33: 'RH', 34: sym('em_dash', '-') };

  const pn = t => `P${t - 20}`;
  const sn = t => SL[t - 60] ?? '?';
  const cn = t => { const ix = t - 70; return ix < 0 || ix > 35 ? '?' : VL[ix % 9] + ' ' + SL[(ix / 9) | 0]; };
  const bv = t => 120 + (t - 120) * 5;
  // fully qualified suit suffix
  const sfx = () => { if (i < tokens.length && isSu(peek())) return ' ' + sn(eat()); return ''; };

  while (i < tokens.length) {
    const t = eat();

    if (t === 1) {
      // START_GAME followed by role
      const r = i < tokens.length ? eat() : 0;
      raw.push({
        ico: em('cards', '*'),
        txt: `Spiel ${sym('middle_dot', '-')} Rolle: <b>${ROLES[r] ?? '?'}</b>`,
        cls: ''
      });

    } else if (t >= 10 && t < 19) {
      raw.push({ ico: em('scroll', '#'), txt: `<b>Stich ${t - 9}</b>`, cls: '' });

    } else if (isP(t)) {
      if (i >= tokens.length) break;
      const act = eat();
      const who = pn(t);

      if (act === 40) {
        let c = '?'; if (i < tokens.length && isCa(peek())) c = cn(eat());
        raw.push({ ico: em('play', '>'), txt: `<b>${who}</b>: spielt <b>${c}</b>`, cls: '' });

      } else if (act === 41) {
        let v = ''; if (i < tokens.length && isBV(peek())) v = bv(eat());
        raw.push({ ico: em('money', '$'), txt: `<b>${who}</b>: Biete <b>${v}</b>`, cls: 'bid-ev' });

      } else if (act === 42) {
        raw.push({ ico: em('blocked', '!'), txt: `<b>${who}</b>: Passe`, cls: '' });

      } else if (act === 43) {
        raw.push({ ico: em('handshake', '+'), txt: `<b>${who}</b>: Gibt Karten`, cls: '' });
        while (i < tokens.length && (isCa(peek()) || peek() === 110)) eat();

      } else if (act === 44) {
        const su = sfx();
        raw.push({ ico: em('trophy', 'T'), txt: `<b>${who}</b>: Trumpf<b>${su}</b>`, cls: 'trump-ev' });

      } else if (act === 45) {
        raw.push({ ico: em('question', '?'), txt: `<b>${who}</b>: Paar?`, cls: '' });

      } else if (act === 46) {
        const su = sfx();
        raw.push({ ico: em('question', '?'), txt: `<b>${who}</b>: Halb?<b>${su}</b>`, cls: '' });

      } else if (act === 47) {
        const su = sfx();
        raw.push({ ico: em('check', '+'), txt: `<b>${who}</b>: Ja, Paar<b>${su}</b>`, cls: 'trump-ev' });

      } else if (act === 48) {
        raw.push({ ico: em('cross', 'x'), txt: `<b>${who}</b>: Nein (kein Paar)`, cls: '' });

      } else if (act === 49) {
        const su = sfx();
        raw.push({ ico: em('check', '+'), txt: `<b>${who}</b>: Ja, Halb<b>${su}</b>`, cls: 'trump-ev' });

      } else if (act === 50) {
        const su = sfx();
        raw.push({ ico: em('cross', 'x'), txt: `<b>${who}</b>: Nein (kein Halb${su})`, cls: '' });
      }

    } else if (t === 51) {
      // Trick won
      if (i < tokens.length && isP(peek())) {
        const w = eat();
        raw.push({ ico: em('trick_win', '*'), txt: `<b>${pn(w)}</b> gewinnt den Stich`, cls: 'trick-win' });
      }
    }
    // unknown tokens: skip silently
  }

  // Render newest-first
  for (let j = raw.length - 1; j >= 0; j--) {
    const { ico, txt, cls } = raw[j];
    const div = document.createElement('div');
    div.className = 'ev' + (cls ? ' ' + cls : '');
    div.innerHTML = `<span class="ev-ico">${ico}</span><span class="ev-txt">${txt}</span>`;
    el.appendChild(div);
  }
}

/** Prepend a single manual log entry (e.g. for button clicks). */
function evLog(ico, txt, cls = '') {
  const el = document.getElementById('evlog');
  const div = document.createElement('div');
  div.className = 'ev' + (cls ? ' ' + cls : '');
  div.innerHTML = `<span class="ev-ico">${ico}</span><span class="ev-txt">${txt}</span>`;
  el.prepend(div);
}

// â”€â”€ 12. AI sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function renderAI(obs) {
  const ct = document.getElementById('ai');
  ct.innerHTML = '';
  const names = ['Du (P0)', 'Links (P1)', 'Partner (P2)', 'Rechts (P3)'];

  for (let s = 0; s < 4; s++) {
    const inf = ainfo[s] || ainfo[String(s)] || {};
    const ctrl = ctrlForSeat(s) || {};
    const probs = inf.probs || [];
    const ent = Number(inf.entropy ?? 0);
    const ep = probs.length ? Math.min(ent / Math.log(Math.max(probs.length, 2)) * 100, 100) : 0;
    const ec = ep < 30 ? 'hl' : ep < 65 ? 'hm' : 'hh';
    const mode = ctrl.mode || inf.controller || 'heuristic';
    const effMode = ctrl.effective_mode || inf.effective_controller || mode;
    const isH = effMode === 'human';
    const cp = pendingSeatCheckpoint[String(s)] || ctrl.checkpoint || inf.checkpoint || 'latest.pt';
    const status = inf.status || '';
    const err = inf.error || '';
    const family = ctrl.model_family || inf.model_family || '';
    const hidden = inf.hidden_summary || {};

    const d = document.createElement('div');
    d.className = 'ais';
    d.innerHTML =
      `<div class="aih">` +
      `<div class="ain">${names[s]}${isH ? ' (Human)' : ''}</div>` +
      `<div style="display:flex;align-items:center;gap:4px">` +
      `<span style="font-size:10px;color:var(--dim)">H</span>` +
      `<div class="hb"><div class="hf ${ec}" style="width:${ep.toFixed(0)}%"></div></div>` +
      `</div>` +
      `</div>` +
      `<div class="prows"></div>` +
      `<div class="ctrl-row">` +
      `<select class="ctrl-mode" data-seat="${s}">` +
      `<option value="human"${mode === 'human' ? ' selected' : ''}>Human</option>` +
      `<option value="heuristic"${mode === 'heuristic' ? ' selected' : ''}>Heuristic</option>` +
      `<option value="model"${mode === 'model' ? ' selected' : ''}>Model</option>` +
      `</select>` +
      `<select class="ctrl-ckpt" data-seat="${s}" ${mode === 'model' ? '' : 'disabled'}></select>` +
      `</div>` +
      `<div class="ctrl-actions">` +
      `<button class="tob reload-model" data-seat="${s}" ${mode === 'model' ? '' : 'disabled'}>Reload</button>` +
      `<button class="tob refresh-ckpt" data-seat="${s}">Refresh</button>` +
      `</div>` +
      `<div class="ai-meta">${status || effMode}${family ? ` | ${family}` : ''}${err ? ` | ${err}` : ''}</div>`;

    const ckSel = d.querySelector('.ctrl-ckpt');
    const ckList = checkpointOptions.length ? checkpointOptions : ['latest.pt'];
    if (!ckList.includes(cp)) ckList.unshift(cp);
    ckList.forEach(name => {
      const op = document.createElement('option');
      op.value = name;
      op.textContent = name;
      if (cp === name) op.selected = true;
      ckSel.appendChild(op);
    });

    d.querySelector('.ctrl-mode')?.addEventListener('change', ev => {
      const seat = +ev.target.dataset.seat;
      const nextMode = ev.target.value;
      const ckpt = d.querySelector('.ctrl-ckpt')?.value || 'latest.pt';
      send({ cmd: 'set_controller', seat, mode: nextMode, checkpoint: ckpt });
    });
    d.querySelector('.ctrl-ckpt')?.addEventListener('change', ev => {
      const seat = +ev.target.dataset.seat;
      const ckpt = ev.target.value;
      pendingSeatCheckpoint[String(seat)] = ckpt;
    });
    d.querySelector('.reload-model')?.addEventListener('click', ev => {
      const seat = +ev.target.dataset.seat;
      const ckpt = d.querySelector('.ctrl-ckpt')?.value || 'latest.pt';
      send({ cmd: 'reload_model', seat, checkpoint: ckpt });
      evLog(em('reload', 'R'), `P${seat} Modell neu geladen: <b>${ckpt}</b>`);
    });
    d.querySelector('.refresh-ckpt')?.addEventListener('click', () => {
      send({ cmd: 'list_checkpoints' });
    });

    // Top-4 action probabilities with chosen-action highlight if available
    const topProbs = [...probs].sort((a, b) => b.prob - a.prob).slice(0, 4);
    const maxP = topProbs[0]?.prob ?? 1;
    const pr = d.querySelector('.prows');
    topProbs.forEach(p => {
      const r = document.createElement('div'); r.className = 'pr';
      const barW = (p.prob / maxP * 70).toFixed(0);
      r.innerHTML =
        `<div class="pl" title="${p.label}">${p.label}</div>` +
        `<div class="pb" style="width:${barW}px"></div>` +
        `<div class="pp">${(p.prob * 100).toFixed(0)}%</div>`;
      if (inf.chosen_label && inf.chosen_label === p.label) r.classList.add('top');
      pr.appendChild(r);
    });

    if (hidden.left || hidden.partner || hidden.right || hidden.hidden_loss) {
      const hsWrap = document.createElement('div');
      hsWrap.className = 'ai-hidden';
      const l = hidden.left || {};
      const p = hidden.partner || {};
      const r = hidden.right || {};
      hsWrap.textContent =
        `HiddenMass L:${Number(l.impossible_mass || 0).toFixed(3)} P:${Number(p.impossible_mass || 0).toFixed(3)} R:${Number(r.impossible_mass || 0).toFixed(3)}`;
      d.appendChild(hsWrap);

      if (hidden.hidden_loss) {
        const h = hidden.hidden_loss;
        const loss = document.createElement('div');
        loss.className = 'ai-hidden';
        loss.textContent = `HiddenLoss T:${Number(h.total || 0).toFixed(3)} Pos:${Number(h.pos_bce || 0).toFixed(3)} Imp:${Number(h.impossible_bce || 0).toFixed(3)}`;
        d.appendChild(loss);
      }
    }

    ct.appendChild(d);
  }

  if (!checkpointOptions.length) send({ cmd: 'list_checkpoints' });
}

// â”€â”€ 13. Debug panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Shows: all player hands (face-up) with belief overlays, completed tricks.

function renderDebugPanel(obs) {
  const p = document.getElementById('dbg-content');
  if (!p) return;
  p.innerHTML = '';
  if (!obs && !Object.keys(allHands).length) return;

  const names = ['Du (P0)', 'P1 Links', 'P2 Partner', 'P3 Rechts'];

  // â”€â”€ Section 1: All hands with face-up cards + belief overlay â”€â”€
  const handsTitle = mk('div', 'dbg-section-title', `${em('play', 'C')} Alle Haende`);
  p.appendChild(handsTitle);

  for (let s = 0; s < 4; s++) {
    const hand = allHands[s] || allHands[String(s)] || [];
    const confirmed = dbgConfirmed[s > 0 ? s - 1 : 0] || [];
    const possible = dbgPossible[s > 0 ? s - 1 : 0] || [];

    const section = mk('div', 'dbg-seat');
    const lbl = mk(
      'div',
      'dbg-seat-lbl',
      names[s] + (hand.length ? ` (${hand.length} Karten)` : ` ${sym('em_dash', '-')} unbekannt`)
    );
    section.appendChild(lbl);

    if (hand.length) {
      const row = mk('div', 'dbg-hand-row');
      hand.forEach(cardIdx => {
        // Build tiny face-up card with belief color border
        const wrapper = mk('div', 'dbg-card-wrap');
        const card = mkCard(cardIdx, '', false);
        const ok = confirmed[cardIdx];
        const pos = possible[cardIdx];
        if (ok) card.style.boxShadow = '0 0 0 2px #22c55e';
        else if (!pos) { card.style.opacity = '0.35'; card.style.boxShadow = '0 0 0 2px #f87171'; }
        wrapper.appendChild(card);
        row.appendChild(wrapper);
      });
      section.appendChild(row);
    } else {
      // Belief grid: show all 36 cards as belief indicators
      const row = mk('div', 'dbg-row');
      for (let idx = 0; idx < 36; idx++) {
        const su = (idx / 9) | 0, va = idx % 9;
        const ok = confirmed[idx];
        const pos = possible[idx];
        const sp = document.createElement('span');
        sp.style.cssText =
          `display:inline-flex;flex-direction:column;align-items:center;font-size:8px;` +
          `padding:1px 2px;border-radius:3px;margin:1px;` +
          `background:${ok ? 'rgba(34,197,94,.2)' : pos ? 'rgba(248,181,0,.08)' : 'rgba(248,113,113,.1)'}`;
        sp.title = `${VL[va]} ${SL[su]}: ${ok ? `${em('check', '+')} Sicher` : pos ? 'Moeglich' : `${em('cross', 'x')} Fehlt`}`;
        sp.appendChild(mkSuitImg(su, 10));
        const vsp = document.createElement('span');
        vsp.style.color = ok ? '#22c55e' : pos ? '#94a3b8' : '#f87171';
        vsp.textContent = VL[va];
        sp.appendChild(vsp);
        row.appendChild(sp);
      }
      section.appendChild(row);
    }
    p.appendChild(section);
  }

  // â”€â”€ Section 2: Completed tricks in order â”€â”€
  // Hidden-state prediction from the model (relative opponents)
  // Set-theory inference diagnostics from symbolic engine.
  if (Object.keys(dbgInference).length) {
    const infTitle = mk('div', 'dbg-section-title', 'Set-Theory Inference');
    p.appendChild(infTitle);
    for (let s = 1; s < 4; s++) {
      const st = dbgInference[String(s)] || dbgInference[s] || {};
      const need = Number(st.need || 0);
      const conf = Number(st.confirmed || 0);
      const poss = Number(st.possible || 0);
      const slack = Number(st.slack || 0);
      p.appendChild(
        mk(
          'div',
          'dbg-seat-lbl',
          `${PNAMES[s]} | confirmed ${conf}/${need} | possible ${poss} | slack ${slack}`
        )
      );
    }
    const singleton = Number(dbgInference.singleton_cards || 0);
    const viol = Number(dbgInference.true_possible_violations || 0);
    const wrong = Number(dbgInference.wrong_confirmed || 0);
    p.appendChild(
      mk(
        'div',
        'dbg-seat-lbl',
        `Global: singletonCards ${singleton} | truePossibleViolations ${viol} | wrongConfirmed ${wrong}`
      )
    );
  }

  if (Object.keys(dbgPredHands).length) {
    const predTitle = mk('div', 'dbg-section-title', 'KI-Hand-Prognose');
    p.appendChild(predTitle);
    if (dbgPredHiddenLoss) {
      const posBce = Number(dbgPredHiddenLoss.pos_bce || 0);
      const impBce = Number(dbgPredHiddenLoss.impossible_bce || 0);
      const total = Number(dbgPredHiddenLoss.total || 0);
      const posAcc = Number(dbgPredHiddenLoss.pos_acc || 0);
      const impMass = Number(dbgPredHiddenLoss.impossible_mass || 0);
      p.appendChild(
        mk(
          'div',
          'dbg-seat-lbl',
          `HiddenLoss total ${total.toFixed(4)} | posBCE ${posBce.toFixed(4)} | impBCE ${impBce.toFixed(4)} | posAcc ${posAcc.toFixed(3)} | impMass ${impMass.toFixed(3)}`
        )
      );
    }

    for (let s = 1; s < 4; s++) {
      const pred = dbgPredHands[String(s)] || dbgPredHands[s] || [];
      const possible = dbgPossible[s - 1] || [];
      const probsSeat = dbgPredCardProbs[s - 1] || [];
      const imp = Number(dbgPredImpossibleMass[String(s)] ?? dbgPredImpossibleMass[s] ?? 0);

      const row = mk('div', 'dbg-trick-row');
      row.appendChild(mk('span', 'dbg-trick-lbl', `${PNAMES[s]} | ImpossibleMass=${imp.toFixed(3)}`));

      pred.forEach(cardIdx => {
        const card = mkCard(cardIdx, '', false);
        const isPossible = !!possible[cardIdx];
        card.style.boxShadow = isPossible ? '0 0 0 2px #22c55e' : '0 0 0 2px #f87171';
        const pr = probsSeat[cardIdx];
        if (pr != null) card.title += ` | p=${(pr * 100).toFixed(1)}%`;
        row.appendChild(card);
      });

      p.appendChild(row);
    }
  }

  const tricks = dbgTricks.length ? dbgTricks : (gs?.info?.tricks || []);
  if (tricks.length) {
    const tTitle = mk('div', 'dbg-section-title', `${em('trick_win', '*')} Gespielte Stiche (${tricks.length})`);
    p.appendChild(tTitle);

    tricks.forEach((trick, ti) => {
      const tRow = mk('div', 'dbg-trick-row');
      const tLbl = mk('span', 'dbg-trick-lbl',
        `#${ti + 1} ${sym('right_arrow', '->')} ${PNAMES[trick.winner] ?? 'P' + trick.winner} (+${trick.points ?? 0})`);
      tRow.appendChild(tLbl);

      const cards = trick.cards || [];
      cards.forEach((cardStr, ci) => {
        const idx = parseCardStr(cardStr);
        if (idx >= 0) {
          const card = mkCard(idx, '', false);
          card.title = PNAMES[trick.players?.[ci] ?? ci];
          tRow.appendChild(card);
        } else {
          tRow.appendChild(mk('span', 'dbg-unknown-card', cardStr));
        }
      });
      p.appendChild(tRow);
    });
  }

  // Section 3: Per-seat policy debug
  const aTitle = mk('div', 'dbg-section-title', 'KI Policy Pro Seat');
  p.appendChild(aTitle);
  for (let s = 0; s < 4; s++) {
    const ai = ainfo[s] || ainfo[String(s)] || {};
    const probs = ai.probs || [];
    const hdr = mk(
      'div',
      'dbg-seat-lbl',
      `${PNAMES[s]} | ${ai.effective_controller || ai.controller || 'n/a'} | H=${Number(ai.entropy || 0).toFixed(2)}`
    );
    p.appendChild(hdr);
    if (!probs.length) continue;
    const allProbs = [...probs].sort((a, b) => b.prob - a.prob).slice(0, 6);
    const maxP = allProbs[0]?.prob ?? 1;
    const pTable = mk('div', 'dbg-policy-table');
    allProbs.forEach((rowP, i) => {
      const row = mk('div', 'dbg-policy-row' + (i === 0 ? ' top' : ''));
      const barW = Math.round(rowP.prob / maxP * 120);
      const pct = (rowP.prob * 100).toFixed(1);
      row.innerHTML =
        `<span class="dbg-plbl" title="${rowP.label}">${rowP.label}</span>` +
        `<span class="dbg-pbar" style="width:${barW}px"></span>` +
        `<span class="dbg-ppct">${pct}%</span>`;
      pTable.appendChild(row);
    });
    p.appendChild(pTable);
  }
  if (!torchOk) {
    p.appendChild(mk('div', 'dbg-no-model', 'Kein Modell geladen - heuristische Steuerung aktiv'));
  }
}

/** Tiny DOM builder helper */
function mk(tag, cls = '', text = '') {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text) el.textContent = text;
  return el;
}

// â”€â”€ 14. Controls & init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

document.getElementById('dbg').addEventListener('change', ev => {
  debugMode = ev.target.checked;
  document.getElementById('dbg-panel').style.display = debugMode ? '' : 'none';
  persistUiPrefs();
  if (gs) { send({ cmd: 'debug_state' }); render(); }
});

function persistUiPrefs() {
  try {
    localStorage.setItem(UI_PREFS_KEY, JSON.stringify({
      debug_mode: debugMode,
      follow_active: followActiveSeat,
      manual_view_seat: preferredManualViewSeat,
    }));
  } catch (_) { }
}

function loadUiPrefs() {
  try {
    const raw = localStorage.getItem(UI_PREFS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      debugMode = parsed?.debug_mode === true;
      followActiveSeat = parsed?.follow_active !== false;
      preferredManualViewSeat = normalizeSeat(parsed?.manual_view_seat ?? 0);
      if (!followActiveSeat) viewSeat = preferredManualViewSeat;
    }
  } catch (_) { }

  const dbg = document.getElementById('dbg');
  if (dbg) dbg.checked = debugMode;
  const dbgPanel = document.getElementById('dbg-panel');
  if (dbgPanel) dbgPanel.style.display = debugMode ? '' : 'none';
  const viewSel = document.getElementById('view-seat');
  if (viewSel) {
    viewSel.value = followActiveSeat ? 'active' : String(preferredManualViewSeat);
  }
}

function clearLocalState() {
  lastTokenLen = 0;
  prevHand.fill(''); prevTrickKey = '';
  for (let s = 0; s < 4; s++) lastKnownHandsBySeat[s] = [];
  _lastTrump = undefined;
  document.getElementById('evlog').innerHTML = '';
  document.getElementById('last-trick-row').style.display = 'none';
  document.getElementById('turn-state').textContent = 'Kein Spiel';
  document.getElementById('bid-area').dataset.ahash = '';
  selectedPassCardsBySeat.forEach(set => set.clear());
}

document.getElementById('bn').addEventListener('click', () => {
  clearLocalState();
  send({ cmd: 'new_game' });
  evLog(em('cards', '*'), 'Neues Spiel gestartet.');
});

document.getElementById('bp').addEventListener('click', () => send({ cmd: 'proceed' }));

document.getElementById('ba').addEventListener('click', () => {
  if (auto) return;
  auto = true;
  const b = document.getElementById('ba');
  b.textContent = 'Auto laeuft';
  b.className = 'btn ba';
  tick();
});

document.getElementById('bq')?.addEventListener('click', () => {
  auto = false;
  clearTimeout(autoTm);
  const b = document.getElementById('ba');
  b.textContent = `${em('play_button', '>')} Auto`;
  b.className = 'btn bs ba';
});

document.getElementById('br')?.addEventListener('click', () => {
  clearLocalState();
  const seedText = (document.getElementById('seed')?.value || '').trim();
  if (seedText === '') {
    send({ cmd: 'new_game' });
    return;
  }
  const seed = Number(seedText);
  if (!Number.isFinite(seed)) {
    evLog(em('cross', 'x'), 'Seed muss eine Zahl sein', 'err-ev');
    return;
  }
  send({ cmd: 'new_game', seed: Math.trunc(seed) });
});

document.getElementById('view-seat')?.addEventListener('change', ev => {
  const value = String(ev.target.value || 'active');
  if (value === 'active') {
    followActiveSeat = true;
    const active = getActiveSeat(gs?.obs);
    if (active >= 0) {
      // Update immediately for responsive UI, then sync with backend.
      viewSeat = active;
      pendingViewSeat = null;
      requestViewSeat(active);
    }
  } else {
    followActiveSeat = false;
    preferredManualViewSeat = normalizeSeat(value);
    // Update immediately for responsive UI, then sync with backend.
    viewSeat = preferredManualViewSeat;
    pendingViewSeat = null;
    requestViewSeat(preferredManualViewSeat);
  }
  persistUiPrefs();
  render();
});

function tick() {
  if (!auto) return;
  if (gs?.done) {
    autoTm = setTimeout(tick, 400);
    return;
  }
  const obs = gs?.obs || null;
  const active = getActiveSeat(obs);
  if (active >= 0 && effectiveMode(active) !== 'human') {
    send({ cmd: 'proceed' });
  }
  autoTm = setTimeout(tick, 450);
}

// Connect â€” do NOT auto-start a new game. The server re-sends state on connect.
loadUiPrefs();
initColumnResizers();
connect();
