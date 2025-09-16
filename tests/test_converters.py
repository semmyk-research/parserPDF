# tests/test_converters.py
# run with pytest tests/.

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from converters.pdf_to_md import PdfToMarkdownConverter
from converters.html_to_md import HtmlToMarkdownConverter
from converters.md_to_pdf import MarkdownToPdfConverter
from converters.extraction_converter import DocumentConverter

@pytest.fixture
def sample_pdf_path():
    # Create a temporary PDF file for testing
    pdf_path = Path("tests/sample.pdf")
    pdf_path.write_bytes(b"%PDF-1.4\nSample PDF content")
    yield pdf_path
    if pdf_path.exists():
        pdf_path.unlink()

@pytest.fixture
def sample_html_path():
    html_path = Path("tests/sample.html")
    html_path.write_text("<html><body><h1>Test</h1><p>Hello World</p></body></html>")
    yield html_path
    if html_path.exists():
        html_path.unlink()

@pytest.fixture
def sample_md_path():
    md_path = Path("tests/sample.md")
    md_path.write_text("# Test\nHello World")
    yield md_path
    if md_path.exists():
        md_path.unlink()

def test_pdf_to_markdown_converter_init():
    converter = PdfToMarkdownConverter()
    assert isinstance(converter, PdfToMarkdownConverter)
    assert hasattr(converter, 'output_dir_string')

@patch('converters.pdf_to_md.Marker')  # Assuming Marker is imported in pdf_to_md.py
def test_pdf_to_markdown_convert_file(mock_marker, sample_pdf_path):
    mock_marker.convert_single.return_value = {"markdown": "# Converted\nContent", "images": []}
    
    converter = PdfToMarkdownConverter()
    result = converter.convert_file(sample_pdf_path)
    
    assert isinstance(result, dict)
    assert "markdown" in result
    assert "filepath" in result
    mock_marker.convert_single.assert_called_once_with(str(sample_pdf_path), prefer_latex=False)

def test_html_to_markdown_converter(sample_html_path):
    converter = HtmlToMarkdownConverter()
    result = converter.batch_convert([sample_html_path])
    
    assert isinstance(result, dict)
    assert Path(sample_html_path.name) in result
    assert result[Path(sample_html_path.name)].startswith("# Test")

def test_markdown_to_pdf_converter(sample_md_path):
    converter = MarkdownToPdfConverter()
    output_dir = Path("tests/output_pdf")
    output_dir.mkdir(exist_ok=True)
    
    pdf_files = converter.batch_convert([sample_md_path], output_dir)
    
    assert isinstance(pdf_files, list)
    if pdf_files:
        pdf_path = pdf_files[0]
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        pdf_path.unlink()
    
    output_dir.rmdir()

@patch('converters.extraction_converter.get_token')
def test_document_converter_login(mock_get_token):
    mock_get_token.return_value = "test_token"
    converter = DocumentConverter()
    assert converter.client.token == "test_token"

def test_pdf_to_markdown_batch_convert(tmp_path):
    # Test batch with multiple files
    pdf1 = tmp_path / "test1.pdf"
    pdf2 = tmp_path / "test2.pdf"
    pdf1.write_bytes(b"%PDF-1.4")
    pdf2.write_bytes(b"%PDF-1.4")
    
    converter = PdfToMarkdownConverter()
    with patch.object(converter, 'convert_file', return_value={"markdown": "test", "filepath": str(pdf1)}):
        results = converter.batch_convert([pdf1, pdf2])
    
    assert len(results) == 2
    assert all("markdown" in res for res in results)