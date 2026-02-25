#!/usr/bin/env python3
"""Atomically replaces the <script> block + adds card-new CSS to index.html"""
import re, pathlib

src = pathlib.Path('ml/ui/index.html').read_text(encoding='utf-8')

# ── 1. Inject .card-new CSS just before </style> ──────────────────────────────
card_new_css = """
    /* entry animation for newly played/drawn cards only */
    @keyframes card-enter {
      from { opacity:0; transform:translateY(-12px) scale(.9) }
      to   { opacity:1; transform:none }
    }
    .card-new { animation: card-enter .22s cubic-bezier(.22,.61,.36,1) both }
"""
src = src.replace('    </style>', card_new_css + '    </style>', 1)

# ── 2. Replace entire <script>…</script> block ────────────────────────────────
new_script = r"""  <script>
    'use strict';
    const SL  = ['Grün','Eichel','Schellen','Herz'];
    const SCOL= ['#15803d','#92400e','#b45309','#dc2626'];
    const VL  = ['6','7','8','9','U','O','K','10','A'];
    const PNAMES = ['Du (P0)','P1 Links','P2 Partner','P3 Rechts'];
    const SUIT_IMGS = ['suits/gruen.svg','suits/eichel.svg','suits/schell.svg','suits/rot.svg'];
    const PORT = location.port || 8765;

    let ws, gs = null, ainfo = {}, hs = new Set(), auto = false, atm;
    let debugMode = false, allHands = {};

    // ── Per-render state for selective animation ──────────────────────────────
    // prevHand[s] = serialized sorted card-idx string of seat s last render
    const prevHand = ['','','',''];
    let prevTrickKey = '';
    let lastTokenLen = 0;
    let bidStepValue = 120;

    // ── Suit image helper (real DOM element — not innerHTML) ──────────────────
    function mkSuitImg(su, sz) {
      const img = document.createElement('img');
      img.src = SUIT_IMGS[su] ?? SUIT_IMGS[0];
      img.width = sz; img.height = sz;
      img.style.cssText = 'display:block;object-fit:contain;pointer-events:none;flex-shrink:0';
      img.draggable = false;
      return img;
    }

    // ── Card builders ─────────────────────────────────────────────────────────
    function mkCard(idx, cls, isNew) {
      const su = (idx / 9) | 0, va = idx % 9, vl = VL[va], c = SCOL[su];
      const e = document.createElement('div');
      e.className = 'card' + (cls?' '+cls:'') + (isNew?' card-new':'');
      e.title = vl + ' ' + SL[su];
      e.dataset.ci = idx;

      const top = document.createElement('div');
      top.style.cssText = 'display:flex;flex-direction:column;align-items:flex-start';
      const tv = document.createElement('div'); tv.className='cv'; tv.style.color=c; tv.textContent=vl;
      top.appendChild(tv); top.appendChild(mkSuitImg(su,12));

      const ctr = document.createElement('div');
      ctr.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%)';
      ctr.appendChild(mkSuitImg(su,28));

      const bot = document.createElement('div'); bot.className='cbot';
      const bv = document.createElement('div'); bv.className='cv'; bv.style.color=c; bv.textContent=vl;
      bot.appendChild(bv); bot.appendChild(mkSuitImg(su,12));

      e.appendChild(top); e.appendChild(ctr); e.appendChild(bot);
      return e;
    }
    function mkFD() { const e=document.createElement('div'); e.className='card fd'; return e; }

    // ── Connection ────────────────────────────────────────────────────────────
    document.getElementById('dbg').addEventListener('change', ev => {
      debugMode = ev.target.checked;
      document.getElementById('dbg-panel').style.display = debugMode ? '' : 'none';
      if (gs) { send({cmd:'debug_state'}); render(); }
    });

    function connect() {
      ws = new WebSocket(`ws://${location.hostname}:${PORT}/ws`);
      dot(0);
      ws.onopen  = () => dot(1);
      ws.onclose = () => { dot(2); setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = ev => {
        const m = JSON.parse(ev.data);
        if (m.type === 'game_state') {
          gs = m.data; ainfo = m.ai_info || {};
          render();
          if (debugMode) send({cmd:'debug_state'});
        } else if (m.type === 'debug_state') {
          allHands = m.hands || {};
          renderDebugPanel(gs?.obs);
        } else if (m.type === 'error') {
          evLog('❌', m.message);
        }
      };
    }
    function send(o)  { if (ws?.readyState===1) ws.send(JSON.stringify(o)); }
    function dot(s)   { document.getElementById('dot').className='dot '+(['','ok','err'][s]||''); }

    // ── Trump display ─────────────────────────────────────────────────────────
    function setTrump(ti) {
      const el = document.getElementById('trump-sym');
      if (el._ti === ti) return;          // no-op if unchanged
      el._ti = ti; el.innerHTML = '';
      if (ti != null) el.appendChild(mkSuitImg(ti, 28)); else el.textContent = '—';
      document.getElementById('trump-lbl').textContent = ti!=null ? SL[ti] : 'Kein Trumpf';
    }

    // ── Active player ─────────────────────────────────────────────────────────
    function getActiveSeat(obs) {
      if (!obs) return -1;
      if ((obs.legal_actions||[]).length > 0) return 0;  // our turn
      const pos = obs.trick_position ?? 0;
      if (pos === 0) return -1;
      return ((obs.current_trick_players?.[0] ?? 0) + pos) % 4;
    }

    // ── Main render ───────────────────────────────────────────────────────────
    function render() {
      if (!gs) return;
      const obs = gs.obs;
      const noGame = document.getElementById('no-game');
      if (!obs) { noGame.style.display=''; return; }
      noGame.style.display = 'none';

      // scores
      document.getElementById('spts').textContent = obs.points_my_team ?? 0;
      document.getElementById('opts').textContent = obs.points_opp_team ?? 0;

      // trump (no-op when unchanged)
      setTrump(obs.trump);

      // hands (selective re-render)
      for (let s=0; s<4; s++) renderHand(s, obs);

      // active player ring
      const active = getActiveSeat(obs);
      for (let s=0; s<4; s++) {
        document.getElementById('p'+s)?.classList.toggle('player-active', s===active);
      }

      // trick (selective re-render + animate only the newest card)
      const trickIdxs    = obs.current_trick_indices  || [];
      const trickPlayers = obs.current_trick_players || [];
      const newTrickKey  = trickIdxs.join(',');
      if (newTrickKey !== prevTrickKey) {
        const tc = document.getElementById('trick');
        tc.innerHTML = '';
        // Which cards are new? Everything not in the previous set
        const prevSet = new Set(prevTrickKey ? prevTrickKey.split(',').map(Number) : []);
        trickIdxs.forEach((ci, i) => {
          const card = mkCard(ci, 'trick', !prevSet.has(ci));
          card.title += ' → ' + PNAMES[trickPlayers[i] ?? i];
          tc.appendChild(card);
        });
        prevTrickKey = newTrickKey;
      }
      document.getElementById('trick-info').textContent =
        `Stich ${obs.trick_number??1} · ${trickIdxs.length}/4 Karten`;

      // bid / action area
      renderBidArea(obs);

      // event log (only if tokens changed)
      renderEventLog(obs);

      // AI info
      renderAI(obs);

      // last trick strip (from end-of-game outcome)
      const info = gs.info;
      if (info?.tricks?.length) {
        const lt = info.tricks[info.tricks.length - 1];
        showLastTrick(lt.cards, lt.winner, lt.points);
      }
    }

    // ── Hand render ───────────────────────────────────────────────────────────
    function renderHand(s, obs) {
      const el = document.getElementById('h'+s);
      if (!el) return;

      if (s===0 && obs.my_hand_indices?.length) {
        const lm = new Map();
        (obs.legal_actions||[]).forEach(la => { if (la.card_idx!=null) lm.set(la.card_idx, la.action_list_idx); });
        const canPlay = hs.size===0 || hs.has(0);
        const newKey  = [...obs.my_hand_indices].sort((a,b)=>a-b).join(',');
        if (newKey !== prevHand[s]) {
          // Hand changed: full re-render, animate only new cards
          const prev = new Set(prevHand[s] ? prevHand[s].split(',').map(Number) : []);
          el.innerHTML = '';
          obs.my_hand_indices.forEach(i => {
            const ai = lm.get(i), legal = canPlay && ai!==undefined;
            const card = mkCard(i, legal?'legal':'', !prev.has(i));
            if (legal) card.addEventListener('click', () => {
              send({cmd:'human_action', action_list_idx: ai});
              evLog('🫵', `Du spielst: <b>${VL[i%9]} ${SL[(i/9)|0]}</b>`);
            });
            el.appendChild(card);
          });
          prevHand[s] = newKey;
        } else {
          // Same cards, just refresh legal highlights
          el.querySelectorAll('.card').forEach(card => {
            const ci = +card.dataset.ci, ai = lm.get(ci);
            card.classList.toggle('legal', canPlay && ai!==undefined);
          });
        }
      } else if (debugMode && allHands[s]?.length) {
        const newKey = [...allHands[s]].sort((a,b)=>a-b).join(',');
        if (newKey !== prevHand[s]) {
          const prev = new Set(prevHand[s] ? prevHand[s].split(',').map(Number) : []);
          const confirmed = obs.confirmed_bitmasks?.[s>0?s-1:0] || [];
          el.innerHTML = '';
          allHands[s].forEach(i => {
            el.appendChild(mkCard(i, confirmed[i]?'debug-confirm':'debug-certain', !prev.has(i)));
          });
          prevHand[s] = newKey;
        }
      } else {
        const n = obs.cards_remaining?.[s] ?? 0;
        // Only rebuild face-down cards if count changed
        if (el.children.length !== n || el.querySelector('.card:not(.fd)')) {
          el.innerHTML = '';
          for (let i=0; i<n; i++) el.appendChild(mkFD());
          prevHand[s] = '';  // marks as face-down (unknown keys)
        }
      }
    }

    // ── Bid area: stepper for Biete, plain buttons for other actions ──────────
    function renderBidArea(obs) {
      const ba = document.getElementById('bid-area');
      if (!obs) { ba.innerHTML=''; return; }
      const canAct = hs.size===0 || hs.has(0);
      const legal  = obs.legal_actions || [];
      const nonCard = legal.filter(la => la.card_idx==null);
      if (!canAct || !nonCard.length) { ba.innerHTML=''; return; }

      const bidActions   = nonCard.filter(la => la.action_token===41);
      const otherActions = nonCard.filter(la => la.action_token!==41);

      // Build a hash to detect if the action set has changed (avoid re-render on same state)
      const aHash = nonCard.map(la=>la.action_list_idx+'_'+la.action_token).join('|');
      if (ba.dataset.ahash === aHash) return;
      ba.dataset.ahash = aHash;
      ba.innerHTML = '';

      if (bidActions.length > 0) {
        const bidVals = bidActions.map(la=>la.bid_value).filter(v=>v!=null).sort((a,b)=>a-b);
        const minB = bidVals[0], maxB = bidVals[bidVals.length-1];
        bidStepValue = Math.max(minB, Math.min(maxB, bidStepValue));
        if (!bidVals.includes(bidStepValue)) {
          bidStepValue = bidVals.reduce((p,c) => Math.abs(c-bidStepValue)<Math.abs(p-bidStepValue)?c:p);
        }

        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:4px;flex-wrap:wrap;justify-content:center;margin-bottom:4px';

        const valDisp = document.createElement('span');
        valDisp.id = 'bid-val-disp';
        valDisp.style.cssText = 'font:700 18px monospace;color:var(--warn);min-width:42px;text-align:center;padding:2px 4px;border:1px solid rgba(210,153,34,.3);border-radius:6px';
        valDisp.textContent = bidStepValue;

        const step = (delta) => {
          ba.dataset.ahash = '';  // invalidate to force re-render
          const idx = bidVals.indexOf(bidStepValue);
          const ni = Math.max(0, Math.min(bidVals.length-1, idx+delta));
          bidStepValue = bidVals[ni];
          valDisp.textContent = bidStepValue;
          confirmBtn.textContent = `Biete ${bidStepValue}`;
        };

        const mkStep = (lbl, d) => {
          const b = document.createElement('button');
          b.className='bid-btn'; b.textContent=lbl; b.style.minWidth='36px';
          b.addEventListener('click', () => step(d));
          return b;
        };

        const step50 = Math.ceil(50/5);
        row.appendChild(mkStep('−50', -step50));
        row.appendChild(mkStep('−5',  -1));
        row.appendChild(valDisp);
        row.appendChild(mkStep('+5',  +1));
        row.appendChild(mkStep('+50', +step50));

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'bid-btn act';
        confirmBtn.textContent = `Biete ${bidStepValue}`;
        confirmBtn.style.marginLeft = '8px';
        confirmBtn.addEventListener('click', () => {
          const la = bidActions.find(a => a.bid_value===bidStepValue);
          if (la) {
            send({cmd:'human_action', action_list_idx: la.action_list_idx});
            evLog('💰', `Du bietest: <b>${bidStepValue}</b>`, 'bid-ev');
          }
        });
        row.appendChild(confirmBtn);
        ba.appendChild(row);
      }

      otherActions.forEach(la => {
        const btn = document.createElement('button');
        let label='', cls='bid-btn';
        switch (la.action_token) {
          case 42: label='Passe';        cls+=' pass'; break;
          case 43: label='Gib ab';       cls+=' pass'; break;
          case 44: label=la.suit_idx!=null? `🏆 Trumpf: ${SL[la.suit_idx]}` :'🏆 Trumpf'; cls+=' act'; break;
          case 45: label='Paar?';        cls+=' act';  break;
          case 46: label=la.suit_idx!=null? `Halb? ${SL[la.suit_idx]}`:'Halb?'; cls+=' act'; break;
          case 47: label='Ja (Paar)';    cls+=' act';  break;
          case 48: label='Nein';         cls+=' pass'; break;
          case 49: label='Ja (Halb)';    cls+=' act';  break;
          case 50: label='Nein';         cls+=' pass'; break;
          default: label=`Aktion ${la.action_token}`;
        }
        btn.className=cls; btn.textContent=label;
        btn.addEventListener('click', () => {
          send({cmd:'human_action', action_list_idx: la.action_list_idx});
          evLog('🫵', `Du: <b>${label}</b>`, 'bid-ev');
        });
        ba.appendChild(btn);
      });
    }

    // ── Last trick strip ──────────────────────────────────────────────────────
    function showLastTrick(cards, winner, pts) {
      const row = document.getElementById('last-trick-row');
      const ltc = document.getElementById('lt-cards');
      if (!cards?.length) { row.style.display='none'; return; }
      // Only update if changed
      const key = cards.join('|') + winner;
      if (row.dataset.key === key) return;
      row.dataset.key = key;
      row.style.display = '';
      document.getElementById('lt-label').textContent =
        `Letzter Stich · ${PNAMES[winner]??'P'+winner} gewinnt (${pts} Pkt.)`;
      ltc.innerHTML = '';
      cards.forEach(cardStr => {
        const idx = parseCardStr(cardStr);
        if (idx >= 0) ltc.appendChild(mkCard(idx, '', false));
        else {
          const sp=document.createElement('span');
          sp.style.cssText='font-size:10px;color:var(--dim);padding:2px';
          sp.textContent=cardStr; ltc.appendChild(sp);
        }
      });
    }

    function parseCardStr(s) {
      const VMAP={'6':0,'7':1,'8':2,'9':3,'U':4,'O':5,'K':6,'10':7,'A':8};
      const SMAP={'Grün':0,'Eichel':1,'Schellen':2,'Herz':3,'Green':0,'Acorns':1,'Bells':2,'Red':3};
      const p=s.trim().split(/\s+/); if(p.length<2) return -1;
      const vi=VMAP[p[0]], si=SMAP[p[1]];
      return (vi==null||si==null) ? -1 : si*9+vi;
    }

    // ── Event log ─────────────────────────────────────────────────────────────
    function renderEventLog(obs) {
      if (!obs) return;
      const tokens = obs.event_tokens || [];
      if (tokens.length === lastTokenLen) return;
      lastTokenLen = tokens.length;

      const el = document.getElementById('evlog');
      el.innerHTML = '';

      // Token constants matching observation.rs
      const P=20, SU=60, CA=70, BV=120;
      const T={START:1,SEP:2,TRICK_WON:51};
      const ROLES={30:'VH',31:'MH',32:'LH',33:'RH',34:'—'};

      const raw=[];
      let i=0;
      const peek=()=>tokens[i];
      const eat=()=>tokens[i++];
      const isP=t=>t>=P&&t<=P+3;
      const isSu=t=>t>=SU&&t<=SU+3;
      const isCa=t=>t>=CA&&t<=CA+35;
      const isBV=t=>t>=BV;
      const pn=t=>`P${t-P}`;
      const sn=t=>SL[t-SU]??'?';
      const cn=t=>{ const ix=t-CA; return ix<0||ix>35?'?': VL[ix%9]+' '+SL[(ix/9)|0]; };
      const bv=t=>120+(t-BV)*5;

      while (i<tokens.length) {
        const t=eat();
        if (t===T.START) {
          if (i<tokens.length) { const r=eat(); raw.push({ico:'🎴',txt:`Spiel · Rolle: <b>${ROLES[r]??'?'}</b>`,cls:''}); }
        } else if (t>=10&&t<19) {
          raw.push({ico:'📋',txt:`<b>── Stich ${t-9} ──</b>`,cls:''});
        } else if (isP(t)) {
          if (i>=tokens.length) break;
          const act=eat();
          if (act===41) {
            let v=''; if(i<tokens.length&&isBV(peek())) v=bv(eat());
            raw.push({ico:'💰',txt:`<b>${pn(t)}</b>: Biete <b>${v}</b>`,cls:'bid-ev'});
          } else if (act===42) {
            raw.push({ico:'⛔',txt:`<b>${pn(t)}</b>: Passe`,cls:''});
          } else if (act===43) {
            raw.push({ico:'🤝',txt:`<b>${pn(t)}</b>: Gibt Karten`,cls:''});
            while(i<tokens.length&&(isCa(peek())||peek()===110)) eat();
          } else if (act===40) {
            let c='?'; if(i<tokens.length&&isCa(peek())) c=cn(eat());
            raw.push({ico:'🃏',txt:`<b>${pn(t)}</b>: spielt <b>${c}</b>`,cls:''});
          } else if (act===44) {
            let su=''; if(i<tokens.length&&isSu(peek())) su=sn(eat());
            raw.push({ico:'🏆',txt:`<b>${pn(t)}</b>: Trumpf <b>${su}</b>`,cls:'trump-ev'});
          } else if (act===45) {
            raw.push({ico:'❓',txt:`<b>${pn(t)}</b>: Paar?`,cls:''});
          } else if (act===46) {
            let su=''; if(i<tokens.length&&isSu(peek())) su=sn(eat());
            raw.push({ico:'❓',txt:`<b>${pn(t)}</b>: Halb? (${su})`,cls:''});
          } else if (act===47) {
            let su=''; if(i<tokens.length&&isSu(peek())) su=sn(eat());
            raw.push({ico:'✅',txt:`<b>${pn(t)}</b>: Ja, Paar ${su}`,cls:'trump-ev'});
          } else if (act===48) {
            raw.push({ico:'❌',txt:`<b>${pn(t)}</b>: Nein (kein Paar)`,cls:''});
          } else if (act===49) {
            let su=''; if(i<tokens.length&&isSu(peek())) su=sn(eat());
            raw.push({ico:'✅',txt:`<b>${pn(t)}</b>: Ja, Halb ${su}`,cls:'trump-ev'});
          } else if (act===50) {
            let su=''; if(i<tokens.length&&isSu(peek())) su=sn(eat());
            raw.push({ico:'❌',txt:`<b>${pn(t)}</b>: Nein (kein Halb ${su})`,cls:''});
          }
        } else if (t===T.TRICK_WON) {
          if (i<tokens.length&&isP(peek())) {
            const w=eat();
            raw.push({ico:'🏅',txt:`<b>${pn(w)}</b> gewinnt den Stich`,cls:'trick-win'});
          }
        }
      }

      // Render newest-first
      for (let j=raw.length-1; j>=0; j--) {
        const {ico,txt,cls}=raw[j];
        const div=document.createElement('div');
        div.className='ev'+(cls?' '+cls:'');
        div.innerHTML=`<span class="ev-ico">${ico}</span><span class="ev-txt">${txt}</span>`;
        el.appendChild(div);
      }
    }

    function evLog(ico, txt, cls='') {
      const el=document.getElementById('evlog');
      const div=document.createElement('div');
      div.className='ev'+(cls?' '+cls:'');
      div.innerHTML=`<span class="ev-ico">${ico}</span><span class="ev-txt">${txt}</span>`;
      el.prepend(div);
    }

    // ── Debug panel ───────────────────────────────────────────────────────────
    function renderDebugPanel(obs) {
      const p=document.getElementById('dbg-content'); if(!p) return;
      p.innerHTML=''; if(!obs) return;
      const names=['P0 (Du)','P1 Links','P2 Partner','P3 Rechts'];
      const SUITS=['Grün','Eichel','Schellen','Herz'];
      const VALS=['6','7','8','9','U','O','K','10','A'];
      for (let s=0; s<4; s++) {
        const confirmed=obs.confirmed_bitmasks?.[s>0?s-1:0]||Array(36).fill(false);
        const possible =obs.possible_bitmasks ?.[s>0?s-1:0]||Array(36).fill(true);
        const d=document.createElement('div'); d.style.marginBottom='10px';
        const lbl=document.createElement('div'); lbl.className='dbg-seat-lbl'; lbl.textContent=names[s];
        d.appendChild(lbl);
        const row=document.createElement('div'); row.className='dbg-row';
        for (let idx=0; idx<36; idx++) {
          const su=(idx/9)|0, va=idx%9;
          const ok=confirmed[idx], pos=possible[idx];
          const sp=document.createElement('span');
          sp.style.cssText=`display:inline-flex;flex-direction:column;align-items:center;font-size:8px;padding:1px 2px;border-radius:3px;margin:1px;background:${ok?'rgba(34,197,94,.15)':pos?'rgba(248,181,0,.08)':'rgba(248,113,113,.1)'}`;
          sp.title=`${VALS[va]} ${SUITS[su]}: ${ok?'✔ Sicher':pos?'Möglich':'✘ Fehlt'}`;
          sp.appendChild(mkSuitImg(su,10));
          const vsp=document.createElement('span'); vsp.style.color=ok?'#22c55e':pos?'#94a3b8':'#f87171'; vsp.textContent=VALS[va];
          sp.appendChild(vsp); row.appendChild(sp);
        }
        d.appendChild(row); p.appendChild(d);
      }
    }

    // ── AI sidebar ────────────────────────────────────────────────────────────
    function renderAI(obs) {
      const ct=document.getElementById('ai'); ct.innerHTML='';
      const names=['Du (P0)','Links (P1)','Partner (P2)','Rechts (P3)'];
      for (let s=0; s<4; s++) {
        const inf=ainfo[s]||{}, probs=inf.probs||[], ent=inf.entropy??0;
        const ep=Math.min(ent/Math.log(Math.max(probs.length,2))*100,100);
        const ec=ep<30?'hl':ep<65?'hm':'hh';
        const isH=hs.has(s);
        const d=document.createElement('div'); d.className='ais';
        d.innerHTML=`<div class="aih"><div class="ain">${names[s]}${isH?' 👤':''}</div>`+
          `<div style="display:flex;align-items:center;gap:4px"><span style="font-size:10px;color:var(--dim)">H</span>`+
          `<div class="hb"><div class="hf ${ec}" style="width:${ep.toFixed(0)}%"></div></div></div></div>`+
          `<div class="prows"></div>`+
          (s>0?`<button class="tob${isH?' on':''}" data-s="${s}">${isH?'✓ Du':'Übernehmen'}</button>`:'');
        const top=[...probs].sort((a,b)=>b.prob-a.prob).slice(0,4);
        const maxP=top[0]?.prob??1;
        const pr=d.querySelector('.prows');
        top.forEach(p=>{
          const r=document.createElement('div'); r.className='pr';
          r.innerHTML=`<div class="pl">${p.label}</div><div class="pb" style="width:${(p.prob/maxP*70).toFixed(0)}px"></div><div class="pp">${(p.prob*100).toFixed(0)}%</div>`;
          pr.appendChild(r);
        });
        d.querySelector('.tob')?.addEventListener('click', ev=>{
          const st=+ev.target.dataset.s; hs.has(st)?hs.delete(st):hs.add(st);
          send({cmd:'set_seat', seat:st, human:hs.has(st)});
          render();
        });
        ct.appendChild(d);
      }
      document.querySelectorAll('.tob[data-s]').forEach(b=>{
        const st=+b.dataset.s;
        b.textContent=hs.has(st)?'✓ Du':'Übernehmen';
        b.className='tob'+(hs.has(st)?' on':'');
      });
    }

    // ── Controls ──────────────────────────────────────────────────────────────
    document.getElementById('bn').addEventListener('click', () => {
      hs.clear(); lastTokenLen=0;
      for(let s=0;s<4;s++) prevHand[s]='';
      prevTrickKey='';
      document.getElementById('evlog').innerHTML='';
      document.getElementById('last-trick-row').style.display='none';
      document.getElementById('bid-area').dataset.ahash='';
      send({cmd:'new_game'});
      evLog('🎴','Neues Spiel gestartet.');
    });
    document.getElementById('bp').addEventListener('click', () => send({cmd:'proceed'}));
    document.getElementById('ba').addEventListener('click', () => {
      auto=!auto;
      const b=document.getElementById('ba');
      b.textContent=auto?'⏸ Pause':'▶ Auto';
      b.className='btn '+(auto?'ba':'bs');
      if(auto) autoStep(); else clearTimeout(atm);
    });
    function autoStep() { if(!auto) return; send({cmd:'proceed'}); atm=setTimeout(autoStep,800); }

    connect();
    // Do NOT auto-start new game on load — server sends existing state on connect.
  </script>"""

# Find script start/end
script_start = src.find('<script>')
script_end   = src.find('</script>', script_start) + len('</script>')
assert script_start >= 0 and script_end > script_start, "Could not find <script> block"

src = src[:script_start] + new_script + src[script_end:]
pathlib.Path('ml/ui/index.html').write_text(src, encoding='utf-8')
lines = src.count('\n')
print(f"Done. {lines} lines written.")
