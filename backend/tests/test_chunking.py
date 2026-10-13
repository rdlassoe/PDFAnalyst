from app.services.chunking import chunk_pages, split_long_paragraph


def _para(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


def test_small_document_single_chunk_keeps_page_range():
    chunks = chunk_pages([(1, "Primer párrafo con contenido suficiente para ser indexado.\n\nSegundo párrafo también largo.")])
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)
    assert "Segundo" in chunks[0].text


def test_chunks_respect_target_and_never_lose_text():
    pages = [(p, "\n\n".join(_para(40, f"p{p}w") for _ in range(6))) for p in range(1, 4)]
    chunks = chunk_pages(pages, target=1000, overlap=150)
    assert len(chunks) > 3
    assert all(len(c.text) <= 1000 for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    for p in range(1, 4):
        assert f"p{p}w0" in joined and f"p{p}w39" in joined


def test_chunks_never_cross_pages_and_cite_one_real_page():
    pages = [(1, _para(60, "a")), (2, _para(60, "b")), (3, _para(60, "c"))]
    chunks = chunk_pages(pages, target=600, overlap=100)
    for c in chunks:
        assert c.page_start == c.page_end
        marker = {1: "a", 2: "b", 3: "c"}[c.page_start]
        other = {"a", "b", "c"} - {marker}
        assert f"{marker}0" in c.text or f"{marker}1" in c.text  # el texto es de la página declarada
        assert not any(f" {o}{i}" in c.text for o in other for i in range(0, 60))  # ni una palabra de otra
    assert [c.page_start for c in chunks] == sorted(c.page_start for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_short_pages_give_one_chunk_each():
    pages = [(p, f"Contenido de la página número {p} con texto suficiente para indexar.") for p in (1, 2, 3)]
    assert [(c.page_start, c.page_end) for c in chunk_pages(pages)] == [(1, 1), (2, 2), (3, 3)]


def test_overlap_repeats_trailing_paragraph_and_terminates():
    paras = [f"Párrafo número {i}. " + "texto " * 30 for i in range(12)]
    chunks = chunk_pages([(1, "\n\n".join(paras))], target=700, overlap=250)
    assert len(chunks) >= 3
    # el último párrafo de un chunk reaparece al inicio del siguiente
    last_para = chunks[0].text.split("\n\n")[-1]
    assert last_para in chunks[1].text


def test_oversized_paragraph_is_split_by_sentences():
    big = " ".join(f"Oración número {i} con algo de texto." for i in range(80))
    pieces = split_long_paragraph(big, 500)
    assert len(pieces) > 4
    assert all(len(p) <= 500 for p in pieces)
    assert pieces[0].startswith("Oración número 0")


def test_unbreakable_token_is_hard_split():
    pieces = split_long_paragraph("x" * 2500, 1000)
    assert [len(p) for p in pieces] == [1000, 1000, 500]


def test_noise_chunks_are_dropped():
    assert chunk_pages([(1, "12")]) == []


def test_empty_pages_are_ignored():
    chunks = chunk_pages([(1, ""), (2, "Contenido real de la segunda página del documento.")])
    assert len(chunks) == 1 and chunks[0].page_start == 2
