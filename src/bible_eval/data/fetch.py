from __future__ import annotations

import json
import urllib.request
from typing import List

from bible_eval.data.loader import Taxonomy

# Public-domain translations available from the scrollmapper/bible_databases repo,
# mapped from our version key to its JSON filename. KJV and ASV are public domain.
SCROLLMAPPER_BASE = "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json"
SCROLLMAPPER_VERSIONS = {
    "kjv": "KJV",
    "asv": "ASV",
}


def source_url(version: str) -> str:
    key = version.lower()
    if key not in SCROLLMAPPER_VERSIONS:
        raise ValueError(
            f"No known public-domain source for version {version!r}. "
            f"Known: {sorted(SCROLLMAPPER_VERSIONS)}"
        )
    return f"{SCROLLMAPPER_BASE}/{SCROLLMAPPER_VERSIONS[key]}.json"


def normalize_scrollmapper(data: dict, taxonomy: Taxonomy) -> List[dict]:
    """Flatten scrollmapper's nested {books:[{name,chapters:[{chapter,verses:[{verse,text}]}]}]}
    into our raw schema: a list of {book, chapter, verse, text}.

    Validates every book name against the taxonomy (fail fast on unknown books)
    and rejects empty verse text, so malformed sources surface immediately.
    """
    books = data.get("books")
    if not isinstance(books, list) or not books:
        raise ValueError("Unexpected source format: missing 'books' list.")

    out: List[dict] = []
    unknown: set[str] = set()
    for bk in books:
        name = str(bk.get("name", "")).strip()
        try:
            taxonomy.book_index(name)
        except KeyError:
            unknown.add(name)
            continue
        for ch in bk.get("chapters", []):
            chapter = int(ch["chapter"])
            for v in ch.get("verses", []):
                text = str(v.get("text", "")).strip()
                if not text:
                    continue  # skip empty verses (e.g. source placeholders)
                out.append(
                    {"book": name, "chapter": chapter, "verse": int(v["verse"]), "text": text}
                )

    if unknown:
        raise ValueError(f"Source book names not in taxonomy (add aliases): {sorted(unknown)}")
    if not out:
        raise ValueError("Source produced no verses.")
    return out


def fetch_version(version: str, taxonomy: Taxonomy, *, url: str | None = None, timeout: int = 60) -> List[dict]:
    """Download and normalize a public-domain translation into our raw verse list."""
    target = url or source_url(version)
    with urllib.request.urlopen(target, timeout=timeout) as resp:  # noqa: S310 - trusted GitHub raw URL
        raw = json.loads(resp.read().decode("utf-8"))
    return normalize_scrollmapper(raw, taxonomy)
