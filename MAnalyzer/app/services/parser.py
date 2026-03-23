"""Raw-text extraction from uploaded files.

All structured interpretation (patient info, lab results, flags) is now
handled by the LLM via ``llm_extractor.py``.  This module's only job is to
turn an uploaded file into a plain-text string suitable for LLM consumption.
"""

import io
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    """Return the raw text content of *filename* for LLM processing."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_text(content)
    if ext in (".csv", ".json", ".txt"):
        return content.decode("utf-8")

    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf_text(content: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "PDF support requires pdfplumber. Install it with: pip install pdfplumber"
        )

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

    full_text = "\n".join(pages)
    if not full_text.strip():
        raise ValueError("Could not extract any text from the PDF.")

    return full_text
