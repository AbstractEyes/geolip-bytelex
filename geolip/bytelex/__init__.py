"""bytelex — universal, model-free relational system over byte information.

Portable library: pure-stdlib core, alphabet-parametric, gram-modular.
The statistics live in the byte corpus itself — no model weights
anywhere — so every ByteLM derivative, present or future, consumes the
same matrix through the same interface.

    from geolip.bytelex import GramSchema, ByteLexicon, build_lexicon
    lex = build_lexicon(chunks)              # any iterable of bytes
    lex.profile(b"understand")               # relational profile
    lex.split_points(b"warehouse")           # corpus-entropy divisions
    project_vocab(lex, "vocab_gpt2.jsonl", "matrix_gpt2.jsonl")

Published artifacts (12 tokenizers, full matrices):
https://huggingface.co/AbstractPhil/geolip-bytelex
"""
from .core import GramSchema, ByteLexicon
from .matrix import (CachedProfiler, alignment_endpoints, build_lexicon,
                     project_vocab, read_vocab)

__version__ = "0.1.0"
__all__ = ["GramSchema", "ByteLexicon", "CachedProfiler", "build_lexicon",
           "project_vocab", "read_vocab", "alignment_endpoints",
           "__version__"]
