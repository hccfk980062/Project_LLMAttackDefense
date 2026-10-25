"""Load attack samples from on-disk JSON datasets.

Handles two schemas:
  * this project's ``data/chinese_attack_set/*.json`` (self-built zh-TW set), and
  * optional external datasets referenced in ``configs/attack.yaml`` (advbench,
    jailbreakbench, ...). Missing external files are tolerated with a warning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List

from .attack_types import AttackSample, AttackType

logger = logging.getLogger(__name__)

CHINESE_ATTACK_SET_DIR = "data/chinese_attack_set"
# Third-party benchmarks among attack.datasets (the rest of that list is this
# project's own zh-TW set, which load_chinese_attack_set() already covers).
EXTERNAL_BENCHMARKS = ("advbench", "jailbreakbench")

_TYPE_ALIASES = {
    "direct_injection": AttackType.DIRECT_INJECTION,
    "indirect_injection": AttackType.INDIRECT_INJECTION,
    "multilingual_obfuscation": AttackType.MULTILINGUAL_OBFUSCATION,
    "jailbreak": AttackType.JAILBREAK,
    "prompt_leaking": AttackType.PROMPT_LEAKING,
}


def _coerce_type(raw: str | None, default: AttackType) -> AttackType:
    if not raw:
        return default
    return _TYPE_ALIASES.get(str(raw).strip().lower(), default)


def _record_to_sample(
    record: dict[str, Any],
    idx: int,
    *,
    source: str,
    default_type: AttackType = AttackType.DIRECT_INJECTION,
) -> AttackSample | None:
    # Payload can live under several keys depending on the dataset.
    payload = (
        record.get("payload")
        or record.get("prompt")
        or record.get("text")
        or record.get("goal")
        or record.get("Goal")          # JailbreakBench raw column name
        or record.get("document_content")
        or record.get("content")
    )
    if not payload or not str(payload).strip():
        return None

    is_document = "document_content" in record and "payload" not in record
    attack_type = _coerce_type(record.get("attack_type"), default_type)
    sample_id = str(record.get("id") or f"{source}_{idx:04d}")

    metadata = {
        k: record[k]
        for k in ("category", "poison_position", "notes", "subcategory", "tags")
        if k in record
    }
    metadata["source"] = source
    if is_document:
        metadata["is_document"] = True

    return AttackSample(
        id=sample_id,
        attack_type=attack_type,
        payload=str(payload),
        malicious_goal=str(record.get("malicious_goal") or record.get("goal") or record.get("Goal") or "unspecified"),
        language=str(record.get("language") or "zh-tw"),
        metadata=metadata,
    )


def load_attack_set_file(
    path: str | Path,
    *,
    default_type: AttackType = AttackType.DIRECT_INJECTION,
    source: str | None = None,
) -> List[AttackSample]:
    """Load a single JSON file (a list of records) into AttackSample objects."""
    path = Path(path)
    if not path.exists():
        logger.warning("Attack set file not found, skipping: %s", path)
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or data.get("records") or []
    if not isinstance(data, list):
        logger.warning("Unexpected JSON shape in %s (expected list), skipping", path)
        return []

    src = source or path.stem
    out: List[AttackSample] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            rec = {"payload": str(rec)}
        sample = _record_to_sample(rec, i, source=src, default_type=default_type)
        if sample is not None:
            out.append(sample)
    return out


def load_chinese_attack_set(
    directory: str | Path = CHINESE_ATTACK_SET_DIR,
) -> List[AttackSample]:
    """Load the self-built Traditional Chinese attack dataset (direct + indirect)."""
    directory = Path(directory)
    samples: List[AttackSample] = []
    file_defaults = {
        "direct_injection.json": AttackType.DIRECT_INJECTION,
        "indirect_injection.json": AttackType.INDIRECT_INJECTION,
    }
    for filename, default_type in file_defaults.items():
        samples.extend(
            load_attack_set_file(
                directory / filename,
                default_type=default_type,
                source=f"chinese_{filename.replace('.json', '')}",
            )
        )
    # Any additional json files dropped into the directory.
    known = set(file_defaults) | {"README.md"}
    for extra in sorted(directory.glob("*.json")):
        if extra.name in known:
            continue
        samples.extend(load_attack_set_file(extra, source=f"chinese_{extra.stem}"))
    return samples


def load_external_attack_datasets(
    attack_config: dict[str, Any] | None,
    *,
    only_enabled: bool = True,
    names: Iterable[str] | None = None,
) -> List[AttackSample]:
    """
    Load datasets listed under ``attack.datasets`` in configs/attack.yaml.

    Every entry is optional: a missing file logs a warning and is skipped so the
    pipeline still runs without advbench / jailbreakbench downloaded.
    ``names`` restricts loading to those dataset keys (e.g. EXTERNAL_BENCHMARKS).
    """
    if not attack_config:
        return []
    datasets = (attack_config.get("attack") or attack_config).get("datasets") or {}
    samples: List[AttackSample] = []
    wanted = set(names) if names is not None else None
    for name, spec in datasets.items():
        if not isinstance(spec, dict) or (wanted is not None and name not in wanted):
            continue
        if only_enabled and not spec.get("enabled", True):
            continue
        path = spec.get("path")
        if not path:
            continue
        default_type = _coerce_type(spec.get("attack_type"), AttackType.DIRECT_INJECTION)
        loaded = load_attack_set_file(path, default_type=default_type, source=name)
        if loaded:
            logger.info("Loaded %d samples from external dataset %r", len(loaded), name)
        samples.extend(loaded)
    return samples


def dedupe_samples(samples: Iterable[AttackSample]) -> List[AttackSample]:
    """Drop samples with a duplicate payload, keeping first occurrence."""
    seen: set[str] = set()
    out: List[AttackSample] = []
    for s in samples:
        key = s.payload.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
