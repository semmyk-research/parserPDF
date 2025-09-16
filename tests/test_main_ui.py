# tests/test_main_ui.py
# run with pytest tests/.

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from main import build_interface  # Wait, main imports from ui, but test main logic
from ui.gradio_ui import convert_batch, build_interface, accumulate_files, clear_state, pdf_files_wrap
from utils.logger import get_logger

logger = get_logger("test_main_ui")

@pytest.fixture
def mock_gradio():
    with patch('gradio.Blocks') as mock_blocks, \
         patch('gradio.Markdown') as mock_md, \
         patch('gradio.Accordion') as mock_accordion, \
         patch('gradio.Dropdown') as mock_dropdown, \
         patch('gradio.Textbox') as mock_textbox, \
         patch('gradio.Slider') as mock_slider, \
         patch('gradio.Checkbox') as mock_checkbox, \
         patch('gradio.Button') as mock_button, \
         patch('gradio.File') as mock_file, \
         patch('gradio.UploadButton') as mock_upload, \
         patch('gradio.State') as mock_state, \
         patch('gradio.Tab') as mock_tab, \
         patch('gradio.JSON') as mock_json, \
         patch('gradio.Files') as mock_files, \
         patch('gradio.Gallery') as mock_gallery:
        yield {
            'Blocks': mock_blocks, 'Markdown': mock_md, 'Accordion': mock_accordion,
            'Dropdown': mock_dropdown, 'Textbox': mock_textbox, 'Slider': mock_slider,
            'Checkbox': mock_checkbox, 'Button': mock_button, 'File': mock_file,
            'UploadButton': mock_upload, 'State': mock_state, 'Tab': mock_tab,
            'JSON': mock_json, 'Files': mock_files, 'Gallery': mock_gallery
        }

def test_build_interface(mock_gradio):
    demo = build_interface()
    assert demo is not None
    # Verify UI components are created
    mock_gradio['Blocks'].assert_called_once_with(title="parserPDF", css=MagicMock())
    mock_gradio['Markdown'].assert_called()  # Title markdown
    mock_gradio['Accordion'].assert_any_call("⚙️ LLM Model Settings", open=False)
    mock_gradio['Tab'].assert_any_call(" 📄 PDF & HTML ➜ Markdown")

def test_convert_batch_no_files():
    result = convert_batch([], 0, "huggingface", "test-model", "fireworks-ai", "", "model-id", 
                          "system", 1024, 0.0, 0.1, False, "token", 
                          "https://router.huggingface.co/v1", "webp", 4, 2, "markdown", 
                          "output_dir", False, None)
    assert "No files uploaded" in result[0]

@patch('ui.gradio_ui.login_huggingface')
@patch('ui.gradio_ui.ProcessPoolExecutor')
@patch('ui.gradio_ui.pdf2md_converter.convert_files')
def test_convert_batch_success(mock_convert, mock_pool, mock_login):
    mock_result = MagicMock()
    mock_convert.return_value = {"filepath": Path("test.md"), "image_path": ["img.jpg"], "markdown": "content"}
    mock_pool.return_value.__enter__.return_value.map.return_value = [mock_result]
    mock_login.return_value = None
    
    pdf_files = [MagicMock(name="test.pdf")]
    result = convert_batch(pdf_files, 1, "huggingface", "test-model", "fireworks-ai", "", "model-id", 
                          "system", 1024, 0.0, 0.1, False, "token", 
                          "https://router.huggingface.co/v1", "webp", 4, 2, "markdown", 
                          "output_dir", False, None)
    
    assert len(result) == 3
    assert "test.md" in result[0]
    assert "img.jpg" in result[2][0]
    mock_pool.assert_called_once()
    mock_convert.assert_called_once_with("test.pdf")

@patch('ui.gradio_ui.ProcessPoolExecutor')
def test_convert_batch_pool_error(mock_pool):
    mock_pool.side_effect = Exception("Pool error")
    pdf_files = [MagicMock(name="test.pdf")]
    result = convert_batch(pdf_files, 1, "huggingface", "test-model", "fireworks-ai", "", "model-id", 
                          "system", 1024, 0.0, 0.1, False, "token", 
                          "https://router.huggingface.co/v1", "webp", 4, 2, "markdown", 
                          "output_dir", False, None)
    assert "Error during ProcessPoolExecutor" in result[0]

def test_accumulate_files():
    # Test initial accumulation
    new_files = [MagicMock(name="/tmp/file1.pdf"), MagicMock(name="/tmp/file2.html")]
    state = []
    updated_state, message = accumulate_files(new_files, state)
    assert len(updated_state) == 2
    assert "/tmp/file1.pdf" in updated_state
    assert "Accumulated 2 file(s)" in message
    
    # Test adding to existing state
    new_files2 = [MagicMock(name="/tmp/file3.pdf")]
    updated_state2, message2 = accumulate_files(new_files2, updated_state)
    assert len(updated_state2) == 3
    assert "Accumulated 3 file(s)" in message2
    
    # Test no new files
    _, message3 = accumulate_files([], updated_state2)
    assert "No new files uploaded" in message3

def test_clear_state():
    result = clear_state()
    assert len(result) == 4
    assert result[0] == []  # cleared file list
    assert result[1] == "Files list cleared."  # message
    assert result[2] == []  # cleared file btn
    assert result[3] == []  # cleared dir btn

def test_pdf_files_wrap():
    # Single file
    single_file = "single.pdf"
    wrapped = pdf_files_wrap(single_file)
    assert isinstance(wrapped, list)
    assert len(wrapped) == 1
    assert wrapped[0] == single_file
    
    # List of files
    files_list = ["file1.pdf", "file2.html"]
    wrapped_list = pdf_files_wrap(files_list)
    assert wrapped_list == files_list
    
    # None input
    assert pdf_files_wrap(None) == [None]

@patch('ui.gradio_ui.os.chdir')
@patch('ui.gradio_ui.Path')
def test_main_launch(mock_path, mock_chdir):
    mock_script_dir = MagicMock()
    mock_path.return_value.resolve.return_value.parent = mock_script_dir
    mock_chdir.return_value = None
    
    # Test main execution path
    with patch('builtins.__name__', '__main__'):
        from main import main  # Assuming main has a main function, or test the if __name__ logic indirectly
        # Since main.py is simple, test the key parts
        demo = MagicMock()
        with patch('ui.gradio_ui.build_interface', return_value=demo):
            with patch('gradio.Interface.launch') as mock_launch:
                # Execute main logic
                import main
                main.main()  # If it has main(), or just import runs it
                
                mock_chdir.assert_called_once_with(mock_script_dir)
                mock_launch.assert_called_once_with(debug=True, show_error=True)