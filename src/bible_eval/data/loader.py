from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bible_eval.utils.verse_id import VerseIdCodec


@dataclass(frozen=True)
class BookDef:
    index: int
    name: str
    aliases: tuple[str, ...]


class Taxonomy:
    def __init__(self, books: list[BookDef]) -> None:
        self.books = books
        self._name_to_index: dict[str, int] = {}
        for b in books:
            self._name_to_index[b.name.casefold()] = b.index
            for a in b.aliases:
                self._name_to_index[a.casefold()] = b.index

    @classmethod
    def from_path(cls, path: str) -> "Taxonomy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        books = [
            BookDef(
                index=int(b["index"]),
                name=str(b["name"]),
                aliases=tuple(str(x) for x in b.get("aliases", [])),
            )
            for b in data["books"]
        ]
        return cls(books=books)

    def book_index(self, book_name: str) -> int:
        key = book_name.strip().casefold()
        if key in self._name_to_index:
            return self._name_to_index[key]
        raise KeyError(f"Unknown book name: {book_name!r}")


@dataclass(frozen=True)
class VerseRecord:
    version: str
    book: str
    chapter: int
    verse: int
    id: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "id": self.id,
            "text": self.text,
            "ref": self.ref,
        }


class VerseDatabase:
    def __init__(self, verses: dict[int, VerseRecord], version: str) -> None:
        self.verses = verses
        self.version = version
        self._ids_sorted = sorted(verses.keys())

    @classmethod
    def from_raw_json(cls, raw_path: str, taxonomy: Taxonomy, version: str) -> "VerseDatabase":
        raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        verses: dict[int, VerseRecord] = {}
        codec = VerseIdCodec()
        for row in raw:
            book = str(row["book"]).strip()
            chapter = int(row["chapter"])
            verse = int(row["verse"])
            text = str(row["text"])
            book_index = taxonomy.book_index(book)
            vid = codec.encode(book_index=book_index, chapter=chapter, verse=verse)
            verses[vid] = VerseRecord(
                version=version,
                book=book,
                chapter=chapter,
                verse=verse,
                id=vid,
                text=text,
            )
        return cls(verses=verses, version=version)

    def all(self) -> list[VerseRecord]:
        return [self.verses[i] for i in self._ids_sorted]
