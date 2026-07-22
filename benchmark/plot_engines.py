"""Shared engine colors, labels, and ordering for benchmark plots.

Canonical paper engines (4-way comparison):
  rpyforth, gforth-fast, vfxforth, swiftforth
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


# Preferred left-to-right / legend order for the four primary engines.
PRIMARY_ENGINES = ("rpyforth", "gforth-fast", "vfxforth", "swiftforth")

ENGINE_COLORS = {
    "rpyforth": "#d62728",
    "rpyforth-c-stkfrag": "#d62728",
    "stkfrag": "#d62728",
    "rpyforth-c": "#8c564b",
    "contiguous": "#8c564b",
    "rpyforth-c-novirt": "#e377c2",
    "novirt": "#e377c2",
    "gforth-fast": "#1f77b4",
    "gforth": "#2ca02c",
    "vfxforth": "#9467bd",
    "swiftforth": "#ff7f0e",
}

# Fallback palette when an engine is unknown.
_FALLBACK_PALETTE = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def normalize_engine(name: str) -> str:
    """Map a binary/config label to a canonical engine id."""
    raw = name.strip()
    base = Path(raw).name.lower()
    # Strip common wrapper / path noise.
    if base.endswith(".sh"):
        base = base[: -len(".sh")]
    aliases = {
        "rpyforth": "rpyforth",
        "rpyforth-c": "rpyforth",
        "rpyforth-c-stkfrag": "rpyforth",
        "rpyforth-c-novirt": "rpyforth",
        "gforth-fast": "gforth-fast",
        "gforth": "gforth",
        "vfxforth": "vfxforth",
        "swiftforth": "swiftforth",
        "sf": "swiftforth",
        "sf64": "swiftforth",
    }
    if base in aliases:
        return aliases[base]
    # Labels may still carry argv tails ("gforth-fast -m 16M").
    head = base.split()[0] if base else base
    return aliases.get(head, head or raw)


def engine_color(engine: str) -> str:
    raw = Path(str(engine).strip()).name.lower()
    if raw.endswith(".sh"):
        raw = raw[: -len(".sh")]
    if raw in ENGINE_COLORS:
        return ENGINE_COLORS[raw]
    key = normalize_engine(engine)
    if key in ENGINE_COLORS:
        return ENGINE_COLORS[key]
    return _FALLBACK_PALETTE[sum(ord(c) for c in key) % len(_FALLBACK_PALETTE)]


def engine_display_name(engine: str) -> str:
    raw = Path(str(engine).strip()).name.lower()
    if raw.endswith(".sh"):
        raw = raw[: -len(".sh")]
    variant_labels = {
        "rpyforth-c-stkfrag": "stkfrag",
        "stkfrag": "stkfrag",
        "rpyforth-c": "contiguous",
        "contiguous": "contiguous",
        "rpyforth-c-novirt": "novirt",
        "novirt": "novirt",
    }
    if raw in variant_labels:
        return variant_labels[raw]
    key = normalize_engine(engine)
    labels = {
        "rpyforth": "rpyforth",
        "gforth-fast": "gforth-fast",
        "gforth": "gforth",
        "vfxforth": "vfxforth",
        "swiftforth": "swiftforth",
    }
    return labels.get(key, key)


def sort_engines(engines: Iterable[str]) -> List[str]:
    """Sort engines with PRIMARY_ENGINES first, then the rest alphabetically."""
    unique: List[str] = []
    seen = set()
    for eng in engines:
        key = normalize_engine(eng)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    primary = [e for e in PRIMARY_ENGINES if e in seen]
    rest = sorted(e for e in unique if e not in PRIMARY_ENGINES)
    return primary + rest


def colors_for_configs(
    config_ids: Sequence[str],
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Map run_shootout config ids (A/B/C/...) to colors via their engine labels."""
    labels = labels or {}
    out: Dict[str, str] = {}
    for cid in config_ids:
        out[cid] = engine_color(labels.get(cid, cid))
    return out
