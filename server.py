"""Bard Lab: a local web playground for the trained model. Standard library only.

  python server.py                 # then open http://localhost:8000
  python server.py --port 9000 --ckpt checkpoints/best.pt

Endpoints:
  GET /                    the single-page UI (web/index.html)
  GET /api/info            model config, checkpoint stats and the training log
  GET /api/generate?...    Server-Sent Events: one event per sampled token, with the
                           model's probability for it and its top alternatives
  GET /api/attention?...   attention weights of one query token over its context
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from model import GPT
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent
MAX_TOKENS = 1000


class App:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = Path(ckpt_path)
        self.model, self.ckpt = GPT.from_checkpoint(ckpt_path)
        self.tok = BPETokenizer.load(self.ckpt["tokenizer"])
        # One request at a time touches the model: attention recording flips per-layer state,
        # and on CPU parallel requests would just split the same cores anyway.
        self.lock = threading.Lock()

    def piece(self, token_id: int) -> str:
        """Display text of one token. Byte fragments of a multi-byte character show as hex."""
        raw = self.tok.token_bytes(token_id)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"<{b:02x}>" for b in raw)

    def info(self) -> dict:
        log_path = self.ckpt_path.parent / "log.jsonl"
        log = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if log_path.exists() else []
        return {"config": self.ckpt["config"], "params": self.model.num_params(),
                "step": self.ckpt["step"], "val_loss": self.ckpt["best_val"],
                "elapsed_s": self.ckpt.get("elapsed"), "log": log}


def _float(q: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(q.get(key, [default])[0])
    except ValueError:
        return default
    return min(max(v, lo), hi) if math.isfinite(v) else default


class Handler(BaseHTTPRequestHandler):
    app: App  # set in main()

    def log_message(self, fmt, *args):  # quieter than the default per-request logging
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._json(self.app.info())
            elif url.path == "/api/generate":
                self.generate(q)
            elif url.path == "/api/attention":
                self.attention(q)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser hung up (e.g. pressed Stop); nothing to clean up

    def generate(self, q: dict) -> None:
        app = self.app
        prompt = q.get("prompt", [""])[0]
        n = int(_float(q, "tokens", 200, 1, MAX_TOKENS))
        temperature = _float(q, "temperature", 0.8, 0.0, 3.0)
        top_k = int(_float(q, "top_k", 0, 0, app.model.cfg.vocab_size)) or None
        top_p = _float(q, "top_p", 1.0, 0.01, 1.0)
        seed = q.get("seed", [""])[0]
        g = torch.Generator().manual_seed(int(seed)) if seed.lstrip("-").isdigit() else None

        ids = app.tok.encode(prompt) or app.tok.encode("\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(event: str, data) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
            self.wfile.flush()

        emit("prompt", [{"id": i, "text": app.piece(i)} for i in ids])
        with app.lock:
            stream = app.model.stream(torch.tensor([ids]), n, temperature=temperature,
                                      top_k=top_k, top_p=top_p, generator=g)
            try:
                for nxt, probs in stream:
                    i = int(nxt)
                    p = probs[0]
                    top_p_vals, top_ids = torch.topk(p, 5)
                    emit("token", {
                        "id": i, "text": app.piece(i), "p": round(float(p[i]), 5),
                        # Entropy in bits: how many "equally likely options" the model weighed.
                        "entropy": round(float(-(p * torch.log2(p.clamp_min(1e-12))).sum()), 3),
                        "top": [{"text": app.piece(int(t)), "p": round(float(v), 5)}
                                for v, t in zip(top_p_vals, top_ids)],
                    })
            finally:
                stream.close()  # resets the model's train/eval state even if the client left
        emit("done", {})

    def attention(self, q: dict) -> None:
        app = self.app
        try:
            ids = [int(x) for x in q.get("ids", [""])[0].split(",") if x]
        except ValueError:
            return self._json({"error": "ids must be comma-separated integers"}, 400)
        vocab, bs = app.model.cfg.vocab_size, app.model.cfg.block_size
        if not ids or any(not 0 <= i < vocab for i in ids):
            return self._json({"error": "bad token ids"}, 400)
        ids = ids[-bs:]  # the query is the last token; it sees at most block_size of context
        with app.lock:
            maps = app.model.attention_maps(torch.tensor([ids]))  # (L, 1, H, T, T)
        last = maps[:, 0, :, -1, :]  # every layer/head's attention from the final token: (L, H, T)
        self._json({"n_ctx": len(ids),
                    "attn": [[[round(float(w), 4) for w in head] for head in layer] for layer in last]})


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the Bard Lab web playground.")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    Handler.app = App(args.ckpt)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Bard Lab running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
