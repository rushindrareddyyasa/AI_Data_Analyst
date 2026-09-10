from ai_data_analyst.rag.chunker import chunk_document


def test_chunk_document():

    text = """
Purpose

This policy defines how regional sales performance is evaluated.

Quarterly Sales Targets

Each region is assigned a quarterly sales target.

A region is considered high-performing when its quarterly sales
exceed its assigned target by more than 10 percent.

Regional Performance Review

Regional performance is evaluated using total sales generated
during the relevant quarter.
""".strip()

    chunks = chunk_document(
        text,
        chunk_size=250,
        chunk_overlap=30,
    )

    assert len(chunks) > 1

    combined_text = "\n".join(chunks)

    assert (
        "A region is considered high-performing"
        in combined_text
    )

    assert (
        "exceed its assigned target by more than 10 percent"
        in combined_text
    )

    for chunk in chunks:
        assert chunk.strip()