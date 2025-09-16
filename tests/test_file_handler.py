# tests/test_file_handler.py
# run with pytest tests/.

import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch

from file_handler.file_utils import (
    collect_pdf_paths, collect_html_paths, collect_markdown_paths,
    process_dicts_data, create_outputdir
)

@pytest.fixture
def temp_dir_with_pdfs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        # Create sample PDF files
        (tmpdir / "doc1.pdf").touch()
        (tmpdir / "subfolder/doc2.pdf").mkdir(parents=True)
        (tmpdir / "subfolder/doc2.pdf").touch()
        (tmpdir / "not_pdf.txt").touch()
        yield tmpdir

@pytest.fixture
def temp_dir_with_html():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "page1.html").touch()
        (tmpdir / "subfolder/page2.htm").mkdir(parents=True)
        (tmpdir / "subfolder/page2.htm").touch()
        (tmpdir / "not_html.md").touch()
        yield tmpdir

@pytest.fixture
def temp_dir_with_md():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        (tmpdir / "note1.md").touch()
        (tmpdir / "subfolder/note2.md").mkdir(parents=True)
        (tmpdir / "subfolder/note2.md").touch()
        (tmpdir / "not_md.pdf").touch()
        yield tmpdir

def test_collect_pdf_paths(temp_dir_with_pdfs):
    paths = collect_pdf_paths(str(temp_dir_with_pdfs))
    assert len(paths) == 2
    assert all(p.suffix.lower() == '.pdf' for p in paths)
    assert Path(str(temp_dir_with_pdfs) / "doc1.pdf") in paths
    assert Path(str(temp_dir_with_pdfs) / "subfolder/doc2.pdf") in paths

def test_collect_pdf_paths_no_pdfs(temp_dir_with_html):
    paths = collect_pdf_paths(str(temp_dir_with_html))
    assert len(paths) == 0

def test_collect_html_paths(temp_dir_with_html):
    paths = collect_html_paths(str(temp_dir_with_html))
    assert len(paths) == 2
    assert all(p.suffix.lower() in ['.html', '.htm'] for p in paths)
    assert Path(str(temp_dir_with_html) / "page1.html") in paths
    assert Path(str(temp_dir_with_html) / "subfolder/page2.htm") in paths

def test_collect_html_paths_no_html(temp_dir_with_pdfs):
    paths = collect_html_paths(str(temp_dir_with_pdfs))
    assert len(paths) == 0

def test_collect_markdown_paths(temp_dir_with_md):
    paths = collect_markdown_paths(str(temp_dir_with_md))
    assert len(paths) == 2
    assert all(p.suffix.lower() == '.md' for p in paths)
    assert Path(str(temp_dir_with_md) / "note1.md") in paths
    assert Path(str(temp_dir_with_md) / "subfolder/note2.md") in paths

def test_collect_markdown_paths_no_md(temp_dir_with_pdfs):
    paths = collect_markdown_paths(str(temp_dir_with_pdfs))
    assert len(paths) == 0

def test_process_dicts_data():
    sample_logs = [
        {"filepath": Path("file1.md"), "markdown": "Content1", "image_path": ["img1.jpg"]},
        {"filepath": Path("file2.md"), "markdown": "Content2", "image_path": []},
        {"error": "Conversion failed for file3"}
    ]
    result = process_dicts_data(sample_logs)
    assert "file1.md" in result
    assert "Content1" in result
    assert "img1.jpg" in result
    assert "Conversion failed" in result

def test_process_dicts_data_empty():
    result = process_dicts_data([])
    assert result == ""

def test_process_dicts_data_invalid():
    with pytest.raises(ValueError):
        process_dicts_data([{"invalid": "data"}])

def test_create_outputdir(tmp_path):
    output_dir = tmp_path / "test_output"
    create_outputdir(str(output_dir))
    assert output_dir.exists()
    assert output_dir.is_dir()

def test_create_outputdir_existing(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    create_outputdir(str(output_dir))
    assert output_dir.exists()
    assert output_dir.is_dir()

@patch('pathlib.Path.mkdir')
def test_create_outputdir_error(mock_mkdir):
    mock_mkdir.side_effect = OSError("Permission denied")
    with pytest.raises(OSError):
        create_outputdir("protected_dir")