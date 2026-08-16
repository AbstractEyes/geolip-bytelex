"""Project a published vocabulary through the published lexicon.
Requires: pip install geolip-bytelex[hub]"""
from geolip.bytelex import project_vocab
from geolip.bytelex.hub import load_lexicon, vocab_path

lex = load_lexicon()                          # byte_lexicon_v1 from HF
summary = project_vocab(lex, vocab_path("gpt2"), "matrix_gpt2_local.jsonl")
print(summary)
