"""Optional Hugging Face artifact loaders (pip install geolip-bytelex[hub]).

The published v1 artifacts live in AbstractPhil/geolip-bytelex:
byte_lexicon_v1/ (corpus tables), vocab_<name>.jsonl, matrix_<name>.jsonl.
"""
from __future__ import annotations

from pathlib import Path

from .core import ByteLexicon

DEFAULT_REPO = "AbstractPhil/geolip-bytelex"
LEXICON_FILES = ("meta.json", "char1.jsonl", "char2.jsonl", "char3.jsonl",
                 "char4.jsonl", "hash9.jsonl", "words.jsonl")


def _dl(repo_id: str, filename: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("pip install geolip-bytelex[hub] for hub loaders") from e
    return hf_hub_download(repo_id, filename, repo_type="model")


def load_lexicon(repo_id: str = DEFAULT_REPO,
                 subdir: str = "byte_lexicon_v1") -> ByteLexicon:
    local = None
    for f in LEXICON_FILES:
        local = _dl(repo_id, f"{subdir}/{f}")
    return ByteLexicon.load(Path(local).parent)


def vocab_path(name: str, repo_id: str = DEFAULT_REPO) -> Path:
    return Path(_dl(repo_id, f"vocab_{name}.jsonl"))


def matrix_path(name: str, repo_id: str = DEFAULT_REPO) -> Path:
    return Path(_dl(repo_id, f"matrix_{name}.jsonl"))
