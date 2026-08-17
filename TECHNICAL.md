# geolip-bytelex — Technical Companion (Beatrix era)

Companion to *Raising Beatrix: A Byte-Level Model's Measured Childhood*
(AbstractPhil with Claude Fable 5 & Claude Opus 5, August 2026).
Deliberately brief there and here: bytelex is the *direction of travel*
stapled to that article's end, and the subject of the next one.

## What it is

A **model-free relational system over byte information**. Model
internals drift as training proceeds; corpus byte statistics do not —
so bytelex plants the reference frame in the corpus and lets every
model (byte-native or tokenized) be measured against it through one
interface. Pure-stdlib core; alphabet-parametric (256 today, by
config); gram-modular (char 1–4 exact, hashed 9-gram, word-grams by
separator predicate — combinations are configuration, not code).

- `core.py` — `GramSchema` (serializable), `ByteLexicon`: gram counts,
  **successor-branching conditional entropy as THE boundary
  functional**, PMI cohesion, split points, save/load/prune. (A
  hand-computed test caught a real normalization bug: corpus-tail
  contexts have no successor; entropy now normalizes over existing
  successors.)
- `matrix.py` — `CachedProfiler` (full-vocab projection in seconds),
  `build_lexicon`, `project_vocab`, `alignment_endpoints(lex, token)` —
  the distillation primitive: where a token's byte expansion sits
  relative to corpus-entropy boundaries.
- `extract.py` — the fleet extractor, in-library as of 2026-08-17:
  token→byte tables from any duck-typed tokenizer. Two families
  (gpt2 byte-unicode; sentencepiece with `<0xNN>` fallback); family
  detection by space-marker census (Ġ vs ▁ — expansion coverage alone
  misdetects, since ascii merges and fallback rows expand under both
  maps); a **mandatory round-trip gate** (a probe sentence must
  reassemble byte-exactly from the table or extraction raises).
- `hub.py` — published artifacts: the corpus lexicon, 13 fleet vocab
  tables (gpt2, cl100k, o200k, llama3, deepseek-v3, mistral-v0.3,
  smollm2, t5, xlm-r, bert, byt5 control, qwen3, qwen2.5), and full
  projections. [Artifacts repo](https://huggingface.co/AbstractPhil/geolip-bytelex).

## First verdicts

- **~27–36% of every generative BPE vocabulary divides at corpus-
  entropy boundaries** — multiple byte-information units wearing one
  token id. The ByT5 control reads exact zeros (instrument validated).
  Consequence: split at the entropy argmax before aligning tokens to a
  byte-native model.
- **The whitespace-convention law**: every generative tokenizer
  attaches whitespace *leading*; byte-native models (measured on
  Beatrix's lexicon census) grow units with *trailing* whitespace.
  Consumers must normalize the convention before any boundary
  comparison — a single off-by-one that otherwise poisons every
  boundary-agreement gauge.

## Where it points

The distillation interface: teacher logits push forward to the shared
256-byte simplex (a pinned frame by construction — no gauge freedom,
no Procrustes step), cohesive tokens align as spans, non-cohesive
tokens split at their entropy argmax, and chunk likelihoods match
between co-boundaries. The first consumer is live: the Beatrix
`bdist-e001` lane (see the
[amoe-lora companion](https://github.com/AbstractEyes/amoe-lora)),
whose alignment gate passed at 1.0000 using the qwen2.5 fleet table
this library extracted. The loss module consuming
`alignment_endpoints` + teacher banks is the next build, and the next
article.
