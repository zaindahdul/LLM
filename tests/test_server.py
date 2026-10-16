import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer

import pytest
import torch

import server
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

TEXT = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n" * 10


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("srv")
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=280)
    tok.save(tmp / "tok.json")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(cfg, verbose=False)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": 7, "elapsed": 1.0,
                "best_val": 5.0, "tokenizer": str(tmp / "tok.json")}, tmp / "best.pt")
    (tmp / "log.jsonl").write_text(json.dumps({"step": 7, "time_s": 1.0, "train_loss": 5.1,
                                               "val_loss": 5.0}) + "\n", encoding="utf-8")

    server.Handler.app = server.App(str(tmp / "best.pt"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode("utf-8")


def sse_events(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_index_page(base_url):
    status, ctype, body = get(base_url + "/")
    assert status == 200 and ctype.startswith("text/html") and "Bard" in body


def test_info(base_url):
    _, _, body = get(base_url + "/api/info")
    info = json.loads(body)
    assert info["step"] == 7 and info["config"]["block_size"] == 16
    assert info["log"][0]["val_loss"] == 5.0


def test_generate_streams_prompt_then_tokens(base_url):
    _, ctype, body = get(base_url + "/api/generate?prompt=To%20be&tokens=40&seed=1&temperature=1")
    assert ctype == "text/event-stream"
    events = sse_events(body)
    kinds = [k for k, _ in events]
    assert kinds[0] == "prompt" and kinds[-1] == "done" and kinds.count("token") == 40
    prompt_text = "".join(t["text"] for t in events[0][1])
    assert prompt_text == "To be"
    for _, tok in events[1:-1]:
        assert 0 <= tok["p"] <= 1 and len(tok["top"]) == 5
        assert tok["top"][0]["p"] >= tok["top"][-1]["p"]


def test_generate_is_reproducible_with_seed(base_url):
    url = base_url + "/api/generate?prompt=the&tokens=20&seed=42"
    assert get(url)[2] == get(url)[2]


def test_generate_tolerates_bad_params(base_url):
    _, _, body = get(base_url + "/api/generate?prompt=&tokens=abc&temperature=nan&top_p=-3")
    assert sse_events(body)[-1][0] == "done"


def test_attention_shape(base_url):
    _, _, body = get(base_url + "/api/attention?ids=" + ",".join(["100"] * 40))
    data = json.loads(body)
    assert data["n_ctx"] == 16  # cropped to block_size
    assert len(data["attn"]) == 2 and len(data["attn"][0]) == 2 and len(data["attn"][0][0]) == 16
    assert abs(sum(data["attn"][1][0]) - 1) < 1e-2


@pytest.mark.parametrize("q", ["ids=", "ids=1,x", "ids=99999"])
def test_attention_rejects_bad_ids(base_url, q):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base_url + "/api/attention?" + q)
    assert e.value.code == 400


def test_unknown_path_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base_url + "/nope")
    assert e.value.code == 404
