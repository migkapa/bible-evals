from bible_eval.utils.verse_id import VerseIdCodec


def test_encode_decode_roundtrip() -> None:
    codec = VerseIdCodec()
    vid = codec.encode(book_index=43, chapter=3, verse=16)
    coord = codec.decode(vid)
    assert coord.book_index == 43
    assert coord.chapter == 3
    assert coord.verse == 16

