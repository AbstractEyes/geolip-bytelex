# geolip-bytelex

**Universal, model-free relational system over byte information.**
Alphabet-parametric gram statistics, corpus-entropy boundaries, and
full-vocabulary token translation matrices — so that *any* ByteLM
derivative can consume supervision from *any* tokenizer-based teacher
through one reusable interface.

- **Model-free.** No weights anywhere. Model internals drift over
  training; corpus byte statistics don't.
- **Alphabet-parametric.** 256 byte values today; the alphabet is a
  schema parameter and may grow or shrink.
- **Gram-modular.** Views are declared, not hardcoded: exact char
  n-grams (1–4), hashed large-n (9-gram class), word-grams via a
  configurable separator predicate. Add trigram/quadgram/9gram/wordgram
  combos as the task requires.
- **Dependency-free core.** Pure stdlib. `huggingface_hub` only behind
  the optional `[hub]` extra.

## Install

```bash
pip install "geolip-bytelex @ git+https://github.com/AbstractEyes/geolip-bytelex"
# with artifact loaders:
pip install "geolip-bytelex[hub] @ git+https://github.com/AbstractEyes/geolip-bytelex"
```

Shares the `geolip` namespace with
[geolip-alephllm](https://github.com/AbstractEyes/alephllm); both can be
installed side by side.

## Quickstart

```python
from geolip.bytelex import GramSchema, build_lexicon, project_vocab

lex = build_lexicon(my_byte_chunks)          # any iterable of bytes
lex.profile(b"understand")                   # relational profile
lex.split_points(b"warehouse")               # where byte-language divides it
project_vocab(lex, "vocab_gpt2.jsonl", "matrix_gpt2.jsonl")
```

Published artifacts (twelve tokenizers — gpt2, cl100k, o200k, Qwen3.8,
DeepSeek-V3, Llama-3.1, Mistral, SmolLM2, T5, XLM-R, BERT, ByT5 —
extracted vocabularies + full matrices + the v1 corpus lexicon):
**https://huggingface.co/AbstractPhil/geolip-bytelex**

```python
from geolip.bytelex.hub import load_lexicon, matrix_path
lex = load_lexicon()                         # byte_lexicon_v1
```

## The three objects

| object | role |
|---|---|
| `GramSchema` | declares alphabet size, gram views, separator predicate — serializable config |
| `ByteLexicon` | corpus statistics: gram counts, successor branching entropy (the boundary functional), adjacent PMI (cohesion); persistable |
| the matrix | every token of a vocabulary → `{hmax, hargmax, pmin, word}` — internal division point, cohesion, word standing |

## Distillation loss primitives

For token-teacher → ByteLM-student alignment:

1. **Endpoints** come from corpus entropy — not from the teacher's
   tokenization and not from any student model's internals.
2. **Cohesive tokens** (low internal entropy) align as single spans.
3. **Non-cohesive tokens** — measured at ~27–36% of every BPE
   vocabulary — split at `hargmax` (`alignment_endpoints()`) before
   matching: they are multiple byte-units wearing one token id.

Validated against a byte-identity control (ByT5): single-byte tokens
show exactly zero internal structure.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

Hand-computed entropy/PMI cases, persistence roundtrips, alphabet
parametricity (non-256 alphabets), projection contract, endpoint
primitives.

Part of the AlephLLM / Mini-Beatrix program. AlephLM is the strongest
current consumer of this structure — but the matrix belongs to the
bytes, not to any one model.


**Technical companion:** [TECHNICAL.md](TECHNICAL.md) — boundary functional, fleet extraction, first verdicts, the distillation interface.
