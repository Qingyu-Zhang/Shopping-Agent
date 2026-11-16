from __future__ import annotations

import json
import re

from .core import stable_id

PASSAGE_CHARS = 900
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def catalog_header(doc):
    fields = [doc.get("title"), doc.get("brand"), doc.get("category"), doc.get("variant"),
              doc.get("sku_id"), doc.get("external_id")]
    return " ".join(str(field) for field in fields if field)


def catalog_sections(doc):
    return [doc.get("description"), *doc.get("features", []), *doc.get("category_path", []),
            *(json.dumps(doc.get(key, {}), ensure_ascii=False) for key in ("specs", "details"))]


def document_text(doc):
    if "passage" in doc:
        return doc["passage"]
    if doc.get("kind") == "review":
        return doc["text"]
    return " ".join(field for field in [catalog_header(doc), *catalog_sections(doc)] if field)


def fragments(text):
    text = text.strip()
    while len(text) > PASSAGE_CHARS:
        window = text[:PASSAGE_CHARS + 1]
        endings = [match.start() for match in SENTENCE_END.finditer(window) if match.start() >= PASSAGE_CHARS // 2]
        end = endings[-1] if endings else window.rfind(" ", PASSAGE_CHARS // 2)
        end = end if end > 0 else PASSAGE_CHARS
        yield text[:end].strip()
        text = text[end:].lstrip()
    if text:
        yield text


def document_passages(doc):
    text = document_text(doc)
    if len(text) <= PASSAGE_CHARS:
        parts = [text]
    elif doc.get("kind") == "review":
        parts = list(fragments(text))
    else:
        header = catalog_header(doc)
        parts = [header]
        for field in catalog_sections(doc):
            if field and field != "{}":
                parts.extend(f"{header}\n{part}" for part in fragments(field))
    for index, part in enumerate(parts):
        key = doc["id"] if not index else stable_id(doc["id"], part, str(index))
        yield {**doc, "id": key, "record_id": doc["id"], "passage": part}
