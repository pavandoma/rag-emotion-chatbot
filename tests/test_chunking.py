from src.utils.chunking import chunk_text


def test_short_text_returns_single_chunk():
    text = "This is a short passage."
    assert chunk_text(text, max_chars=500) == [text]


def test_empty_text_returns_no_chunks():
    assert chunk_text("", max_chars=500) == []


def test_long_text_splits_into_multiple_chunks():
    sentence = "This is a sentence that takes up some space. "
    text = sentence * 30
    chunks = chunk_text(text, max_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)


def test_chunks_preserve_overlap():
    sentence = "Sentence number {}. "
    text = "".join(sentence.format(i) for i in range(20))
    chunks = chunk_text(text, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
