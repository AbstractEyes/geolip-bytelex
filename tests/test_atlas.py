"""Atlas tests: determinism, dead-head defaults, hand-computed caveats, consensus, coverage,
spectra, and the mint screens — all on toy vocabularies (no downloads)."""
import math

from geolip.bytelex.atlas import WeightField, cell_index, cell_bytes, load_vocab_rows

TOY = [(0, b" the"), (1, b" then"), (2, b" them"), (3, b" they"), (4, b"ring"), (5, b"rink")]


def build(scheme="flat"):
    return WeightField.from_rows(TOY, scheme)


def test_dead_head_default_zero():
    f = build()
    dead = cell_index(1, 2, 3)
    assert f.cell_strength(dead) == 0.0
    assert f.caveat(dead) is None
    assert dead not in f.strength


def test_paths_edges_end_mass_hand_computed():
    f = build()
    c_the = cell_index(*b"the")                      # inside " the*" x4 and terminal for " the"
    assert f.cell_strength(c_the) == 4.0
    e = f.edges[c_the]
    assert e == {ord("n"): 1.0, ord("m"): 1.0, ord("y"): 1.0}
    assert f.end_mass[c_the] == 1.0                  # " the" ends there
    h, top, close = f.caveat(c_the)
    assert abs(h - math.log2(3)) < 1e-9              # three equal continuations
    assert abs(top - 1 / 3) < 1e-9
    assert abs(close - 0.25) < 1e-9                  # 1 end vs 3 continuations


def test_forced_continuation_caveat():
    f = build()
    h, top, close = f.caveat(cell_index(*b"rin"))    # rin -> g or k, no ends
    assert abs(h - 1.0) < 1e-9 and abs(top - 0.5) < 1e-9 and close == 0.0


def test_rank_scheme_orders_by_id():
    f = WeightField.from_rows([(0, b"abcd"), (100, b"abce")], "rank")
    e = f.edges[cell_index(*b"abc")]
    assert e[ord("d")] > e[ord("e")]


def test_consensus_sums_fields():
    f1, f2 = build("flat"), build("flat")
    c = WeightField.consensus([f1, f2])
    assert c.cell_strength(cell_index(*b"the")) == 8.0
    assert c.caveat(cell_index(*b"rin"))[1] == 0.5


def test_coverage_gap():
    f1 = WeightField.from_rows([(0, b"abc")], "flat")
    f2 = WeightField.from_rows([(0, b"xyz")], "flat")
    assert f1.coverage_gap(f2) == {cell_index(*b"abc")}


def test_spectrum_shape_and_dead_marking():
    f = build()
    sp = f.spectrum(b"theq")
    assert len(sp) == 2
    assert sp[0]["strength"] == 4.0 and sp[0]["caveat"] is not None
    assert sp[1]["strength"] == 0.0 and sp[1]["caveat"] is None    # "heq" is a dead head


def test_mint_screens_and_determinism():
    rows = [(i, b" " + bytes(w)) for i, w in enumerate(
        [b"stone", b"stole", b"stork", b"blane", b"bloke", b"crane", b"crest", b"plume", b"grath", b"snide"])]
    f = WeightField.from_rows(rows, "flat")
    known = {"stone", "stole"}
    a = f.mint_words(4, seed=7, length_range=(4, 5), known_words=known, avoid={"crane"})
    b = f.mint_words(4, seed=7, length_range=(4, 5), known_words=known, avoid={"crane"})
    assert a == b                                     # deterministic under the seed
    assert all(w not in known and w != "crane" for w in a)
    assert all(w.isalpha() and 4 <= len(w) <= 5 for w in a)
    # pairwise flanked-trigram sharing respects the cap
    def flanked(w):
        s = " " + w + " "
        return {s[i:i + 3] for i in range(len(s) - 2)}
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            assert len(flanked(a[i]) & flanked(a[j])) <= 1


def test_mint_prefix_pinning():
    rows = [(i, b" " + w) for i, w in enumerate([b"brant", b"brisk", b"brome", b"bruce", b"crash"])]
    f = WeightField.from_rows(rows, "flat")
    out = f.mint_words(2, seed=3, length_range=(4, 5), prefix=b"br", max_shared_trigrams=2)
    assert out and all(w.startswith("br") for w in out)


def test_load_vocab_rows_skips_specials(tmp_path):
    p = tmp_path / "vocab_toy.jsonl"
    p.write_text('{"id":0,"hex":"5b5041445d","text":"[PAD]","is_special":true}\n'
                 '{"id":1,"hex":"20746865","text":" the","is_special":false}\n')
    rows = load_vocab_rows(p)
    assert rows == [(1, b" the")]


def test_cell_roundtrip():
    for b in (b"abc", b"\xe4\xb8\xad", b"\x00\xff\x10"):
        assert cell_bytes(cell_index(*b)) == b


def test_mint_lexicon_program_structure_and_determinism():
    words = [b"stone", b"stole", b"stork", b"blane", b"bloke", b"blimp", b"crane", b"crest",
             b"crumb", b"plume", b"plank", b"grath", b"snide", b"swill", b"flint", b"flame",
             b"glide", b"gloam", b"chart", b"chime", b"drape", b"trove", b"bruce", b"brisk"]
    f = WeightField.from_rows([(i, b" " + w) for i, w in enumerate(words)], "flat")
    kw = dict(n_easy=3, n_trap=3, pair_prefixes=(b"st", b"bl"), seed=11, length_range=(4, 5),
              easy_band=(0.0, 1.0), trap_band=(0.0, 1.0), known_words={"stone"}, avoid={"crane"})
    a = f.mint_lexicon_program(**kw)
    b = f.mint_lexicon_program(**kw)
    assert a == b                                           # deterministic under the seed
    allw = a["easy"] + a["trap"] + [w for p in a["pairs"] for w in p]
    assert len(a["easy"]) == 3 and len(a["trap"]) == 3 and len(a["pairs"]) == 2
    assert len(set(allw)) == len(allw)                      # no duplicates across bands
    assert "stone" not in allw and "crane" not in allw      # novelty + blocklist screens
    for (x, y), pf in zip(a["pairs"], (b"st", b"bl")):
        assert x.startswith(pf.decode()) and y.startswith(pf.decode())

    def flanked(w):
        s = " " + w + " "
        return {s[i:i + 3] for i in range(len(s) - 2)}
    solo = a["easy"] + a["trap"]                            # cross-cap holds among non-pair words
    for i in range(len(solo)):
        for j in range(i + 1, len(solo)):
            assert len(flanked(solo[i]) & flanked(solo[j])) <= 1
