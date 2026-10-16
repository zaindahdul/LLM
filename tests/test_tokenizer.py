import random

import pytest

from tokenizer import BPETokenizer, SPLIT_PATTERN, _merge

# Small, varied training text so the tests train in well under a second.
TRAIN_TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. I'll say it's 1234567 times, we've seen 3.14!\n"
    "naïve café — 你好世界 — привет — مرحبا — 🎉🚀👨‍👩‍👧 snake_case_name\n"
) * 20

TRICKY_STRINGS = [
    "",
    "a",
    " ",
    "\n\n\n",
    "   leading and trailing spaces   ",
    "tabs\tand\r\nwindows newlines",
    "Hello, world!",
    "emoji 😀🎉🚀 and ZWJ family 👨‍👩‍👧‍👦 and flags 🇺🇸🇯🇵",
    "skin tones 👍🏽👋🏿",
    "combining: é vs é, Zalgo: Z̵̡̢a̴l̶g̷o̸",
    "CJK 你好世界 こんにちは 안녕하세요",
    "RTL مرحبا بالعالم שלום",
    "math ∑∫√∞ ≠ ≤ and superscripts x² y³",
    "code: def f(x_1, y__2): return {'k': [1, 2]}  # comment",
    "numbers 1234567890 3.14159 -42 1e10",
    "contractions: I'll you've they're it's don't",
    "null\x00byte and bell\x07 and DEL\x7f",
    "private use  and 4-byte \U0001F9E0 and max \U0010FFFF",
    "weird spaces: nbsp em　ideographic",
    "_",
    "___",
    "'",
    "'s",
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(TRAIN_TEXT, vocab_size=400)
    return t


@pytest.mark.parametrize("text", TRICKY_STRINGS)
def test_roundtrip_tricky(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_untrained_is_raw_bytes():
    t = BPETokenizer()  # no merges: ids should be exactly the UTF-8 bytes
    s = "héllo 🎉"
    assert t.encode(s) == list(s.encode("utf-8"))
    assert t.decode(t.encode(s)) == s


def _random_text(rng: random.Random, length: int) -> str:
    chars = []
    for _ in range(length):
        r = rng.random()
        if r < 0.5:
            chars.append(rng.choice("abcdefghijklmnopqrstuvwxyz ABCXYZ.,!?'\n\t_0123456789"))
        else:
            # Any code point except surrogates (U+D800..U+DFFF), which are not valid
            # Unicode scalar values and cannot be UTF-8 encoded by any tokenizer.
            cp = rng.randrange(0, 0x110000 - 0x800)
            if cp >= 0xD800:
                cp += 0x800
            chars.append(chr(cp))
    return "".join(chars)


def test_roundtrip_random_unicode(tok):
    rng = random.Random(0)
    for _ in range(300):
        text = _random_text(rng, rng.randrange(0, 80))
        assert tok.decode(tok.encode(text)) == text


def test_pretokenizer_covers_every_character():
    # If the regex skipped any character, it would be silently dropped by encode().
    import re
    regex = re.compile(SPLIT_PATTERN)
    rng = random.Random(1)
    for text in TRICKY_STRINGS + [_random_text(rng, 100) for _ in range(200)]:
        assert "".join(regex.findall(text)) == text


def test_merge_helper():
    assert _merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert _merge([5, 5, 5], (5, 5), 9) == [9, 5]  # non-overlapping, left to right
    assert _merge([], (1, 2), 9) == []


def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer()
    t.train("aaabdaaabac", vocab_size=257)  # classic BPE example: "aa" occurs most often
    assert list(t.merges) == [(ord("a"), ord("a"))]
    assert t.vocab[256] == b"aa"


def test_training_compresses(tok):
    sample = "To be, or not to be, that is the question"
    assert len(tok.encode(sample)) < len(sample.encode("utf-8")) / 2


def test_vocab_size(tok):
    assert tok.vocab_size == 400
    assert all(0 <= i < 400 for i in tok.encode(TRAIN_TEXT))


def test_training_is_deterministic(tok):
    t2 = BPETokenizer()
    t2.train(TRAIN_TEXT, vocab_size=400)
    assert t2.merges == tok.merges


def test_save_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab == tok.vocab
    for s in TRICKY_STRINGS:
        assert loaded.encode(s) == tok.encode(s)


def test_decode_partial_utf8_does_not_crash():
    t = BPETokenizer()
    ids = t.encode("🎉")  # 4 bytes
    assert t.decode(ids[:2]) == "�"  # half an emoji decodes to the replacement char


def test_vocab_too_small_raises():
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", vocab_size=100)
