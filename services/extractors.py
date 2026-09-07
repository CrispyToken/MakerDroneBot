from pathlib import Path
from io import BytesIO
from config import TEXT_EXTENSIONS

_docling_converter = None


def _get_docling_converter():
    """Lazy singleton: DocumentConverter holds the loaded layout/table models,
    so it must not be recreated per file."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _extract_with_docling(data: bytes, suffix: str) -> str:
    """Convert a document to structured Markdown via Docling.

    Unlike pypdf/python-docx, Docling runs layout analysis and TableFormer,
    so headings, reading order, and tables survive as Markdown instead of
    being flattened into a raw text stream.
    """
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        converter = _get_docling_converter()
        result = converter.convert(tmp_path)
        return result.document.export_to_markdown().strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_pdf(data: bytes) -> str:
    return _extract_with_docling(data, ".pdf")


def extract_docx(data: bytes) -> str:
    return _extract_with_docling(data, ".docx")


def extract_html(data: bytes) -> str:
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(data, "lxml")
    except Exception:
        soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_epub(data: bytes) -> str:
    import tempfile, ebooklib, os
    from ebooklib import epub
    from bs4 import BeautifulSoup
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            book = epub.read_epub(tmp_path, options={"ignore_ncx": True})
        except TypeError:
            book = epub.read_epub(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    parts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    parts.append(text)
            except Exception:
                continue
    return "\n\n".join(parts).strip()


def extract_text_from_file(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        return extract_pdf(data)
    if ext == ".docx":
        return extract_docx(data)
    if ext in {".html", ".htm"}:
        return extract_html(data)
    if ext == ".epub":
        return extract_epub(data)
    raise ValueError(f"Unsupported file extension: {ext}")