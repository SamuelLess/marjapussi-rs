"""
WebSocket + HTTP server for the Marjapussi AI UI.
Run:  python ml/ui_server.py [--checkpoint CKPT] [--port 8765]
Open: http://localhost:8765
"""
import argparse, asyncio, json, math, sys
from pathlib import Path

import aiohttp
from aiohttp import web

ROOT = Path(__file__).parent.parent
ML   = Path(__file__).parent
UI   = ML / "ui"
CHECKPOINT_DIR = ML / "checkpoints"
RUNS_DIR       = ML / "runs"

sys.path.insert(0, str(ML))

from env import MarjapussiEnv, obs_to_tensors

try:
    import torch, torch.nn.functional as F
    from model import MarjapussiNet
    TORCH_OK = True
except ImportError as e:
    TORCH_OK = False
    print(f"[warn] PyTorch not available: {e} — demo mode")


# ── Game manager ──────────────────────────────────────────────────────────────
class GameManager:
    def __init__(self, checkpoint=None):
        self.env  = None
        self.obs  = None
        self.done = True
        self.info = {}
        self.human_seats: set = set()
        self.model = None
        if TORCH_OK:
            self.model = MarjapussiNet(); self.model.eval()
            p = checkpoint or (CHECKPOINT_DIR / "latest.pt")
            if Path(p).exists():
                self.model.load_state_dict(torch.load(str(p), map_location="cpu"), strict=False); print(f"[model] loaded {p}")

    def new_game(self):
        if self.env:
            try: self.env.close()
            except: pass
        bin_path = ROOT / "target" / "debug" / "ml_server.exe"
        if not bin_path.exists():
            bin_path = ROOT / "target" / "debug" / "ml_server"
        if not bin_path.exists():
            raise FileNotFoundError(
                "ml_server binary not found — run: cargo build --bin ml_server")
        self.env  = MarjapussiEnv(pov=0)
        self.env.env_binary = str(bin_path)          # ensure right path
        self.obs  = self.env.reset()
        self.done = self.env.done

    def state(self):
        return {"obs": self.obs, "done": self.done, "info": self.info,
                "human_seats": list(self.human_seats)}

    def ai_info(self):
        if not TORCH_OK or self.model is None or self.obs is None:
            return {}
        try:
            tensors = obs_to_tensors(self.obs)
            with torch.no_grad():
                logits, _, _, _ = self.model(tensors)
            probs = F.softmax(logits[0], dim=-1).tolist()
            legal = self.obs.get("legal_actions", [])
            entropy = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
            SUITS = ['Grün', 'Eichel', 'Schellen', 'Herz']
            VALS  = ['6', '7', '8', '9', 'U', 'O', 'K', '10', 'A']
            ATOK  = {40:'spielt', 41:'Biete', 42:'Passe', 43:'Gib',
                     44:'Trumpf', 45:'Paar?', 46:'Halb?',
                     47:'Ja-Paar', 48:'Nein-Paar', 49:'Ja-Halb', 50:'Nein-Halb'}
            prob_list = []
            for i, la in enumerate(legal):
                if i >= len(probs): break
                tok = la.get("action_token", 40)
                base = ATOK.get(tok, f"Akt{tok}")
                # Fully-qualify label with card, bid value or suit
                if la.get("card_idx") is not None:
                    c = la["card_idx"]
                    lbl = f"{base} {VALS[c % 9]} {SUITS[c // 9]}"
                elif la.get("bid_value") is not None:
                    lbl = f"{base} {la['bid_value']}"
                elif la.get("suit_idx") is not None:
                    lbl = f"{base} {SUITS[la['suit_idx']]}"
                else:
                    lbl = base
                prob_list.append({"label": lbl, "prob": probs[i]})
            return {0: {"probs": prob_list, "entropy": entropy}}
        except Exception as e:
            print(f"[ai_info error] {e}")
            return {}

    def debug_state(self):
        """Returns all hands and completed tricks for debug overlay."""
        result = {"hands": {}, "tricks": []}
        if self.obs is None:
            return result
        # All hands (now reliably provided by Rust in the obs JSON)
        hands = self.obs.get("all_hands", [])
        if hands:
            result["hands"] = {str(i): hand for i, hand in enumerate(hands)}
        # Completed tricks from outcome info
        if hasattr(self, "info") and self.info:
            result["tricks"] = self.info.get("tricks", [])
        # Include belief bitmasks from obs for symbolic reasoning display
        result["confirmed_bitmasks"] = self.obs.get("confirmed_bitmasks", [])
        result["possible_bitmasks"]  = self.obs.get("possible_bitmasks", [])
        result["event_tokens"]       = self.obs.get("event_tokens", [])
        return result

    def ai_step(self):
        if self.done or self.obs is None: return
        legal = self.obs.get("legal_actions", [])
        if not legal: return
        chosen = 0
        if TORCH_OK and self.model is not None:
            try:
                tensors = obs_to_tensors(self.obs)
                with torch.no_grad(): logits, _, _, _ = self.model(tensors)
                probs = F.softmax(logits[0], dim=-1)
                chosen = int(torch.multinomial(probs, 1).item())
                chosen = min(chosen, len(legal) - 1)
            except: pass
        self.obs, self.done, self.info = self.env.step(legal[chosen]["action_list_idx"])

    def step(self, action_list_idx):
        self.obs, self.done, self.info = self.env.step(action_list_idx)

    def debug_pass(self, card_indices):
        self.obs, self.done, self.info = self.env.debug_pass(card_indices)

    def load_checkpoint(self, name):
        if not TORCH_OK: return
        p = CHECKPOINT_DIR / f"{name}.pt"
        if not p.exists(): p = Path(name)
        if p.exists():
            self.model.load_state_dict(torch.load(str(p), map_location="cpu"), strict=False)
            self.model.eval(); print(f"[model] reloaded {p}")


game    = None
clients: set = set()

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request); clients.add(ws)
    # Send current state on connect
    if game.obs is not None:
        await ws.send_str(json.dumps({"type":"game_state","data":game.state(),"ai_info":game.ai_info()}))
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try: await handle(ws, json.loads(msg.data))
            except Exception as e: await ws.send_str(json.dumps({"type":"error","message":str(e)}))
        elif msg.type == aiohttp.WSMsgType.ERROR: break
    clients.discard(ws); return ws

async def handle(ws, cmd):
    act = cmd.get("cmd")
    if act == "new_game":
        game.new_game()
    elif act == "proceed":
        if not game.done: game.ai_step()
    elif act == "human_action":
        game.step(cmd.get("action_list_idx", 0))
    elif act == "debug_pass":
        game.debug_pass(cmd.get("card_indices", []))
    elif act == "set_seat":
        s = cmd.get("seat", 0)
        if cmd.get("human"): game.human_seats.add(s)
        else: game.human_seats.discard(s)
    elif act == "load_checkpoint":
        game.load_checkpoint(cmd.get("checkpoint", "latest"))
    elif act == "debug_state":
        ds = game.debug_state()
        await ws.send_str(json.dumps({"type": "debug_state", **ds}))
        return  # don't broadcast full state for debug
    await broadcast()

async def broadcast():
    msg = json.dumps({"type":"game_state","data":game.state(),"ai_info":game.ai_info()})
    dead = set()
    for c in clients:
        try: await c.send_str(msg)
        except: dead.add(c)
    clients.difference_update(dead)

async def watch_log(_app):
    asyncio.create_task(_watch_log())

async def _watch_log():
    while True:
        p = RUNS_DIR / "latest" / "log.jsonl"
        if p.exists():
            try:
                lines = p.read_text().splitlines()
                for line in reversed(lines[-20:]):
                    entry = json.loads(line)
                    if entry.get("event") in ("update", "eval"):
                        stats = {"game": entry.get("game"), "stage": entry.get("stage", 0),
                                 "loss": entry.get("losses", {}).get("total"),
                                 "win_rate": entry.get("win_rate")}
                        msg = json.dumps({"type":"train_stats","data":stats})
                        for c in list(clients):
                            try: await c.send_str(msg)
                            except: clients.discard(c)
                        break
            except: pass
        await asyncio.sleep(2)

async def index(request):
    return web.FileResponse(UI / "index.html")

async def cleanup(app):
    """Gracefully terminate the Rust subprocess on Ctrl+C."""
    if game and game.env:
        try: game.env.close()
        except: pass

def main():
    global game
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    game = GameManager(args.checkpoint)
    try:
        game.new_game()
        print("[game] started OK")
    except Exception as e:
        print(f"[warn] Could not start game: {e}")
        print("[warn] Build ml_server first: cargo build --bin ml_server")

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/ui", UI)        # serve CSS/JS
    app.router.add_static("/suits", UI / "suits")  # serve suit SVG images
    app.on_startup.append(watch_log)
    app.on_cleanup.append(cleanup)

    print(f"\n🃏  Marjapussi KI  →  http://localhost:{args.port}")
    print(f"   PyTorch: {'✓' if TORCH_OK else '✗ (demo mode)'}")
    if TORCH_OK and game.model: print(f"   Params:  {game.model.param_count():,}")
    print()
    web.run_app(app, port=args.port, print=False)

if __name__ == "__main__":
    main()
