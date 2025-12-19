from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerseCoord:
    book_index: int
    chapter: int
    verse: int


class VerseIdCodec:
    """
    Canonical ID encoding:
      id = (book_index * 1_000_000) + (chapter * 1_000) + verse
    """

    def encode(self, book_index: int, chapter: int, verse: int) -> int:
        if book_index <= 0:
            raise ValueError("book_index must be >= 1")
        if chapter <= 0 or verse <= 0:
            raise ValueError("chapter and verse must be >= 1")
        if chapter >= 1000 or verse >= 1000:
            raise ValueError("chapter and verse must be < 1000")
        return (book_index * 1_000_000) + (chapter * 1_000) + verse

    def decode(self, verse_id: int) -> VerseCoord:
        if verse_id <= 0:
            raise ValueError("verse_id must be positive")
        book_index = verse_id // 1_000_000
        rem = verse_id % 1_000_000
        chapter = rem // 1_000
        verse = rem % 1_000
        return VerseCoord(book_index=book_index, chapter=chapter, verse=verse)
