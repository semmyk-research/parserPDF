# tests/test_conversion.py
# run with pytest tests/.

from pathlib import Path
from converters.pdf_to_md import PdfToMarkdownConverter
from converters.md_to_pdf import MarkdownToPdfConverter

def test_sample_pdf():
    pdf = Path("tests/sample.pdf")
    converter = PdfToMarkdownConverter()
    md = converter.convert(pdf)
    assert isinstance(md, str) and len(md) > 0

def test_markdown_to_pdf(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\nThis is a test.")
    conv = MarkdownToPdfConverter()
    pdf_path = conv.convert(md_file)
    assert pdf_path.exists() and pdf_path.suffix == ".pdf"
