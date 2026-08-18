"""Extractor tests — synthetic duck-typed tokenizers, no downloads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geolip.bytelex.extract import (_gpt2_unicode_to_byte, coverage, detect_family,
                                    roundtrip_gate, token_byte_table,
                                    write_vocab_jsonl)


def _b2u():
    return {v: k for k, v in _gpt2_unicode_to_byte().items()}


class FakeGPT2Tok:
    """Byte-level BPE over a tiny vocab: 256 byte tokens + 2 merges +
    1 special, GPT-2 unicode-remap storage."""
    def __init__(self):
        b2u = _b2u()
        self._toks = [ "".join(b2u[b] for b in bytes([i]))
                       for i in range(256)]
        self._toks.append("".join(b2u[b] for b in b"an"))
        self._toks.append("".join(b2u[b] for b in b"swer"))
        self._toks.append("<|end|>")
        self.all_special_tokens = ["<|end|>"]

    def __len__(self):
        return len(self._toks)

    def convert_ids_to_tokens(self, i):
        return self._toks[i]

    def encode(self, text, add_special_tokens=False):
        out = []
        data = text.encode("utf-8")
        i = 0
        while i < len(data):
            if data[i:i + 4] == b"swer":
                out.append(257)
                i += 4
            elif data[i:i + 2] == b"an":
                out.append(256)
                i += 2
            else:
                out.append(data[i])
                i += 1
        return out


class FakeSPTok:
    def __init__(self):
        self._toks = (["▁the", "▁answer", "answer", "."]
                      + [f"<0x{i:02X}>" for i in range(256)]
                      + ["<s>"])
        self.all_special_tokens = ["<s>"]

    def __len__(self):
        return len(self._toks)

    def convert_ids_to_tokens(self, i):
        return self._toks[i]

    def encode(self, text, add_special_tokens=False):
        out = []
        for word in text.split(" "):
            if word == "the":
                out.append(0)
            elif word == "answer":
                out.append(1)
            else:
                for b in (" " + word).encode("utf-8"):
                    out.append(4 + b)
        return out


def test_family_detection():
    assert detect_family(FakeGPT2Tok()) == "gpt2"
    assert detect_family(FakeSPTok()) == "sentencepiece"


def test_gpt2_roundtrip_and_table():
    tk = FakeGPT2Tok()
    rows = token_byte_table(tk)
    assert rows[256]["hex"] == b"an".hex()
    assert rows[258]["is_special"]
    roundtrip_gate(tk, rows, probe="an answer!")


def test_sentencepiece_expansions():
    tk = FakeSPTok()
    rows = token_byte_table(tk, family="sentencepiece")
    assert bytes.fromhex(rows[0]["hex"]) == b" the"
    assert bytes.fromhex(rows[4]["hex"]) == bytes([0])
    assert rows[-1]["is_special"]


def test_write_summary(tmp_path):
    s = write_vocab_jsonl(FakeGPT2Tok(), tmp_path / "v.jsonl",
                          probe="an answer!")
    assert s["family"] == "gpt2" and s["n_tokens"] == 259
    assert s["n_special"] == 1




class FakeWPTok:
    def __init__(self):
        self._toks = (["the", "answer", "##er", "##s", ".", "an"]
                      + ["[CLS]", "[SEP]", "[unused0]"])
        self.all_special_tokens = ["[CLS]", "[SEP]"]

    def __len__(self):
        return len(self._toks)

    def convert_ids_to_tokens(self, i):
        return self._toks[i]

    def encode(self, text, add_special_tokens=False):
        out = []
        for w in text.lower().split(" "):
            if w == "the":
                out.append(0)
            elif w == "answers":
                out.extend([1, 3])
            elif w == "answer":
                out.append(1)
            elif w == "an":
                out.append(5)
        return out


def test_wordpiece_family_and_gate():
    tk = FakeWPTok()
    assert detect_family(tk) == "wordpiece"
    rows = token_byte_table(tk, family="wordpiece")
    assert bytes.fromhex(rows[0]["hex"]) == b" the"
    assert bytes.fromhex(rows[2]["hex"]) == b"er"
    assert rows[8]["is_special"]          # [unused0]
    roundtrip_gate(tk, rows, probe="The Answers", mode="normalized")


def test_coverage_gauge():
    tk = FakeWPTok()
    rows = token_byte_table(tk, family="wordpiece")
    c = coverage(rows, b" the answers the")
    assert c["coverage"] > 0.9
    c2 = coverage(rows, b"zzzzqqqq")
    assert c2["coverage"] == 0.0


if __name__ == "__main__":
    import tempfile
    test_family_detection()
    test_gpt2_roundtrip_and_table()
    test_sentencepiece_expansions()
    test_wordpiece_family_and_gate()
    test_coverage_gauge()
    with tempfile.TemporaryDirectory() as d:
        test_write_summary(Path(d))
    print("extract tests: 4/4 PASS")
