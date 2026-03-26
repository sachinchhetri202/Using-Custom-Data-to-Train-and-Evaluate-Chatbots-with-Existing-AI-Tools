from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


NAV_MARKERS = {
    "WSU Home",
    "eWeber",
    "Campus Directory",
    "Canvas",
    "Majors, Minors & Programs",
    "Nondiscrimination & Accessibility",
}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_id(*parts: str, length: int = 16) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.replace("\u00a0", " ")
    return text.strip()


def is_navigation_noise(text: str) -> bool:
    if not text:
        return True
    marker_count = sum(marker in text for marker in NAV_MARKERS)
    bullet_count = text.count("•")
    return marker_count >= 2 or bullet_count >= 12


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
