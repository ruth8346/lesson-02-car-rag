"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
import re
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """One chunk per "## " section, each starting with a "<document title> › <section heading>" line.

    The text before the first "## " goes in a chunk whose path line is just the title.
    A section with no text (e.g. a scanned document that was never extracted) makes no chunk.
    """
    preamble, *sections = re.split(r"^## +", doc.text, flags=re.MULTILINE)
    preamble = re.sub(r"^# .*$", "", preamble, count=1, flags=re.MULTILINE).strip()

    parts = [(None, preamble)]
    for section in sections:
        heading, _, body = section.partition("\n")
        parts.append((heading.strip(), body.strip()))

    chunks = []
    for heading, body in parts:
        if not body:
            continue
        path = f"{doc.title} › {heading}" if heading else doc.title
        meta = dict(doc.meta, section=heading or "")
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, f"{path}\n{body}", meta))
    return chunks
