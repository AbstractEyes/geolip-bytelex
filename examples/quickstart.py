"""Build a lexicon from your own bytes and read a token's standing."""
from geolip.bytelex import build_lexicon

corpus = [open(__file__, "rb").read()]      # any iterable of bytes
lex = build_lexicon(corpus, prune_min_count=0)

for tok in (b"lexicon", b"build_le", b"the "):
    print(tok, "->", lex.profile(tok))
    print("   splits at:", lex.split_points(tok))
