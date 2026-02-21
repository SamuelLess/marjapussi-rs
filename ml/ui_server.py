"""
WebSocket + HTTP server for the Marjapussi AI visualization UI.

Usage:
    python ml/ui_server.py [--checkpoint path] [--port 8765]

Open: http://localhost:8765
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import math
from pathlib import Path

import aiohttp
from aiohttp import web
import asyncio

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    from model import MarjapussiNet
    from env import MarjapussiEnv, obs_to_tensors
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed — running in demo mode (random actions only)")

RUNS_DIR = Path(__file__).parent / "runs"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
UI_DIR = Path(__file__).parent / "ui"

# ── Game manager ──────────────────────────────────────────────────────────────

class GameManager:
    def __init__(self, checkpoint_path=None):
        self.env = None
        self.model = None
        self.human_seats = set()    # seat indices controlled by human
        self.checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE:
            return
        self.model = MarjapussiNet()
        self.model.eval()
        path = self.checkpoint_path or (CHECKPOINT_DIR / "latest.pt")
        if Path(path).exists():
            self.model.load_state_dict(torch.load(str(path), map_location='cpu'))
            print(f"Loaded model from {path}")

    def new_game(self):
        if self.env:
            try: self.env.close()
            except: pass
        self.env = MarjapussiEnv(pov=0)
        self.obs = self.env.reset()
        self.done = False
        self.info = {}

    def current_state(self):
        return {
            'obs': self.obs,
            'done': self.done,
            'info': self.info,
            'human_seats': list(self.human_seats),
        }

    def ai_info_for(self, obs):
        """Compute model policy info for all seats."""
        seat_info = {}
        if not TORCH_AVAILABLE or self.model is None:
            return seat_info
        # Only compute for POV seat (0) — other seat views would need separate obs
        try:
            tensors = obs_to_tensors(obs)
            with torch.no_grad():
                logits, _, _ = self.model(tensors)
            logits = logits[0]
            probs = F.softmax(logits, dim=-1).tolist()
            legal = obs.get('legal_actions', [])
            entropy = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)

            from env import encode_legal_actions
            from model import ACTION_LABELS as _  # reuse token labels

            ACTION_TOKEN_LABELS = {
                40: 'Play', 41: 'Bid', 42: 'Stop bid', 43: 'Pass cards',
                44: 'Trump', 45: 'Ask pair', 46: 'Ask half',
                47: 'Yes pair', 48: 'No pair', 49: 'Yes half', 50: 'No half',
            }

            prob_list = []
            for i, la in enumerate(legal):
                if i >= len(probs):
                    break
                tok = la.get('action_token', 40)
                label = ACTION_TOKEN_LABELS.get(tok, f'Act {tok}')
                if la.get('card_idx') is not None:
                    c = la['card_idx']
                    suits = ['g', 'e', 's', 'r']
                    vals = ['6','7','8','9','U','O','K','10','A']
                    label += f" {vals[c%9]}♠{suits[c//9]}"
                prob_list.append({'label': label, 'prob': probs[i]})

            seat_info[0] = {'probs': prob_list, 'entropy': entropy}
        except Exception as e:
            print(f"ai_info error: {e}")
        return seat_info

    def step(self, action_list_idx: int):
        if not self.env or self.done:
            return
        self.obs, self.done, self.info = self.env.step(action_list_idx)

    def ai_step(self):
        """Let the AI pick and apply an action for the current seat."""
        if not self.env or self.done:
            return

        legal = self.obs.get('legal_actions', [])
        if not legal:
            return

        if TORCH_AVAILABLE and self.model is not None:
            try:
                tensors = obs_to_tensors(self.obs)
                with torch.no_grad():
                    logits, _, _ = self.model(tensors)
                probs = F.softmax(logits[0], dim=-1)
                action_pos = torch.multinomial(probs, 1).item()
                action_pos = min(action_pos, len(legal) - 1)
                chosen = legal[action_pos]
            except:
                chosen = legal[0]
        else:
            # Random fallback
            import random
            chosen = random.choice(legal)

        self.step(chosen['action_list_idx'])

    def load_checkpoint(self, name: str = 'latest'):
        if not TORCH_AVAILABLE:
            return
        path = CHECKPOINT_DIR / f"{name}.pt"
        if not path.exists() and not name.endswith('.pt'):
            path = CHECKPOINT_DIR / f"{name}"
        if path.exists():
            self.model.load_state_dict(torch.load(str(path), map_location='cpu'))
            self.model.eval()
            print(f"Loaded checkpoint {path}")


# ── WebSocket handler ──────────────────────────────────────────────────────────

game = None
clients = set()

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)

    # Send initial state
    if game.env:
        await ws.send_str(json.dumps({
            'type': 'game_state',
            'data': game.current_state(),
            'ai_info': game.ai_info_for(game.obs),
        }))

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                cmd = json.loads(msg.data)
                await handle_cmd(ws, cmd)
            except Exception as e:
                await ws.send_str(json.dumps({'type': 'error', 'message': str(e)}))
        elif msg.type == aiohttp.WSMsgType.ERROR:
            break

    clients.discard(ws)
    return ws


async def handle_cmd(ws, cmd):
    action = cmd.get('cmd')

    if action == 'new_game':
        game.new_game()

    elif action == 'proceed':
        if not game.done:
            game.ai_step()

    elif action == 'human_action':
        idx = cmd.get('action_list_idx', 0)
        game.step(idx)

    elif action == 'set_seat':
        seat = cmd.get('seat', 0)
        human = cmd.get('human', False)
        if human:
            game.human_seats.add(seat)
        else:
            game.human_seats.discard(seat)

    elif action == 'load_checkpoint':
        name = cmd.get('checkpoint', 'latest')
        game.load_checkpoint(name)

    # Broadcast updated state to all clients
    state_msg = json.dumps({
        'type': 'game_state',
        'data': game.current_state(),
        'ai_info': game.ai_info_for(game.obs) if game.obs else {},
    })
    for client in list(clients):
        try:
            await client.send_str(state_msg)
        except:
            clients.discard(client)


# ── Training log watcher ───────────────────────────────────────────────────────

async def watch_training_log():
    """Tail ml/runs/latest/log.jsonl and push training stats to clients."""
    while True:
        log_path = RUNS_DIR / "latest" / "log.jsonl"
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                for line in reversed(lines[-20:]):
                    entry = json.loads(line.strip())
                    if entry.get('event') in ('update', 'eval'):
                        stats = {
                            'game': entry.get('game'),
                            'stage': entry.get('stage', 0),
                            'loss': entry.get('losses', {}).get('total'),
                            'win_rate': entry.get('win_rate'),
                        }
                        msg = json.dumps({'type': 'train_stats', 'data': stats})
                        for client in list(clients):
                            try:
                                await client.send_str(msg)
                            except:
                                clients.discard(client)
                        break
            except:
                pass
        await asyncio.sleep(2)


# ── HTTP static file server ────────────────────────────────────────────────────

async def index_handler(request):
    return web.FileResponse(UI_DIR / 'index.html')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global game

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    game = GameManager(checkpoint_path=args.checkpoint)
    game.new_game()

    app = web.Application()
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/', index_handler)
    app.router.add_static('/ui', UI_DIR)

    async def start_watchers(app):
        asyncio.create_task(watch_training_log())

    app.on_startup.append(start_watchers)

    print(f"\n🃏  Marjapussi AI UI")
    print(f"   Open: http://localhost:{args.port}")
    print(f"   PyTorch: {'available' if TORCH_AVAILABLE else 'NOT INSTALLED'}")
    if TORCH_AVAILABLE and game.model:
        print(f"   Model params: {game.model.param_count():,}")
    print()

    web.run_app(app, port=args.port, print=False)


if __name__ == '__main__':
    main()
