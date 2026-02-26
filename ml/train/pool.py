import queue
import threading
import torch
import torch.nn.functional as F

from env import MarjapussiEnv
from .utils import Log

class BatchInferenceServer:
    """
    Accumulates inference requests from all worker threads and fires one
    GPU forward pass per batch. Expects tensors to be passed in to
    parallelize CPU-heavy conversion.
    """
    def __init__(self, model, device, max_batch: int = 128, timeout: float = 0.005, greedy: bool = False):
        self.model   = model
        self.device  = device
        self.max_batch = max_batch
        self.timeout   = timeout
        self.greedy    = greedy
        self._req  = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def infer(self, tensor_obs) -> tuple[int, float, float]:
        """Block until inference result is available. Returns (action_idx, value, log_prob). Thread-safe."""
        evt  = threading.Event()
        slot = [0, 0.0, 0.0]
        self._req.put((tensor_obs, evt, slot))
        evt.wait()
        return slot[0], slot[1], slot[2]

    def _serve(self):
        while not self._stop.is_set():
            items = []
            try:
                items.append(self._req.get(timeout=self.timeout))
                while len(items) < self.max_batch:
                    try: items.append(self._req.get_nowait())
                    except queue.Empty: break
            except queue.Empty:
                continue
            if not items:
                continue
            self._run_batch(items)

    def _run_batch(self, items):
        dev = self.device
        tensors_list = [it[0] for it in items]
        try:
            B = int(len(tensors_list))
            # Quick collate for inference only
            max_s = int(max(int(t["token_ids"].shape[1]) for t in tensors_list))
            max_a = int(max(int(t["action_feats"].shape[1]) for t in tensors_list))
            
            # Batch together and MOVE TO DEVICE
            obs_a = {k: torch.cat([t["obs_a"][k] for t in tensors_list], 0).to(dev)
                     for k in tensors_list[0]["obs_a"]}
            tok = torch.zeros((B, max_s), dtype=torch.long, device=dev)
            tmask = torch.ones((B, max_s), dtype=torch.bool, device=dev)
            for i, t in enumerate(tensors_list):
                L = int(t["token_ids"].shape[1])
                tok[i,:L] = t["token_ids"][0].to(dev); tmask[i,:L] = t["token_mask"][0].to(dev)
            
            af = torch.zeros((B, max_a, 51), device=dev)
            am = torch.ones((B, max_a), dtype=torch.bool, device=dev)
            for i, t in enumerate(tensors_list):
                A = t["action_feats"].shape[1]
                af[i,:A] = t["action_feats"][0].to(dev); am[i,:A] = t["action_mask"][0].to(dev)

            with torch.no_grad():
                logits, _, _, value_pred = self.model({
                    "obs_a":       obs_a,
                    "token_ids":   tok, "token_mask": tmask,
                    "action_feats": af, "action_mask": am,
                })
            probs = F.softmax(logits, dim=-1).cpu()
            values = value_pred.cpu()
            for i, (_, evt, slot) in enumerate(items):
                if self.greedy:
                    act = int(torch.argmax(probs[i]).item())
                else:
                    act = int(torch.multinomial(probs[i], 1).item())
                slot[0] = act
                slot[1] = float(values[i].item())
                slot[2] = float(torch.log(probs[i, act] + 1e-8).item())
                evt.set()
        except Exception as e:
            Log.error(f"Batch inference failed: {e}")
            for _, evt, slot in items:
                slot[0] = 0; evt.set()

    def stop(self):
        self._stop.set(); self._thread.join(timeout=1)


class EnvPool:
    """Pool of persistent MarjapussiEnv instances."""
    def __init__(self, size: int, include_labels: bool = False):
        self.envs = queue.Queue()
        for _ in range(size):
            self.envs.put(MarjapussiEnv(include_labels=include_labels))

    def get(self) -> MarjapussiEnv:
        return self.envs.get()

    def put(self, env: MarjapussiEnv):
        self.envs.put(env)

    def close(self):
        while not self.envs.empty():
            self.envs.get().close()
