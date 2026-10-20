from __future__ import annotations

from .models import Chunk, Document


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split on paragraph boundaries into chunks of about `size` chars with `overlap` carry-over.

    Paragraphs longer than `size` are hard-split, so a chunk never exceeds `size + overlap + 2`.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")

    pieces: list[str] = []
    for para in (p.strip() for p in doc.content.split("\n\n")):
        while len(para) > size:
            pieces.append(para[:size])
            para = para[size - overlap :]
        if para:
            pieces.append(para)

    texts: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            texts.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n\n{piece}" if current else piece
    if current:
        texts.append(current)

    return [
        Chunk(id=f"{doc.id}-{i}", document_id=doc.id, content=text, metadata=doc.metadata)
        for i, text in enumerate(texts)
    ]
