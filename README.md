---
title: parser2md - PDF & HTML parser to markdown
emoji: 📚
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
python_version: 3.12
command: python main.py
app_file: main.py
hf_oauth: true
#oauth_scopes: [read-access]
hf_oauth_scopes: [read-access, inference-api]
license: mit
pinned: true
short_description: PDF & HTML parser to markdown
#models: [meta-llama/Llama-4-Maverick-17B-128E-Instruct, openai/gpt-oss-120b, openai/gpt-oss-20b, ]
models: 
  - meta-llama/Llama-4-Maverick-17B-128E-Instruct
  - openai/gpt-oss-120b, openai/gpt-oss-20b
  - vikp/surya_det3
  - vikp/surya_rec2
  - vikp/surya_tablerec
  - datalab-to/surya_layout
  - datalab-to/surya_tablerec
  - datalab-to/texify
  - datalab-to/ocr_error_detection
  - datalab-to/inline_math_det0
  - datalab-to/line_detector0
  - xiaoyao9184/surya_text_detection
  - xiaoyao9184/surya_text_recognition
  - xiaoyao9184/surya_table_recognition
  - xiaoyao9184/surya_texify
  - xiaoyao9184/surya_layout
  - xiaoyao9184/surya_ocr_error_detection
  - xiaoyao9184/surya_inline_math_detection]
tags: [markdown, PDF, parser, converter, extractor]
#preload_from_hub: [https://huggingface.co/datalab-to/surya_layout, https://huggingface.co/datalab-to/surya_tablerec, huggingface.co/datalab-to/line_detector0, https://huggingface.co/tarun-menta/ocr_error_detection/blob/main/config.json]
owner: research-semmyk
#---
#
#[Project]
#---
#title: parser2md - PDF & HTML parser to markdown
#emoji: \U0001F4C4📝📑
#colorFrom: yellow
#colorTo: purple
#sdk: gradio
#python_version: 3.12
#sdk_version: 5.44.1
#app_file: main.py
#command: python main.py
#models:
#  - meta-llama/Llama-4-Maverick-17B-128E-Instruct
#  - openai/gpt-oss-120b
#pinned: false
#license: mit
#name: parser2md
#short_description: PDF & HTML parser to markdown
version: 0.1.0
readme: README.md
requires-python: ">=3.12"
#dependencies: []
#preload_from_hub:
#  - https://huggingface.co/datalab-to/surya_layout
#  - https://huggingface.co/datalab-to/surya_tablerec
#  - huggingface.co/datalab-to/line_detector0
#  - https://huggingface.co/tarun-menta/ocr_error_detection/blob/main/config.json
#owner: research-semmyk
## Model list
#[
#    "datalab/models/text_recognition/2025_08_29",
#    "datalab/models/layout/2025_02_18",
#    "datalab/models/table_recognition/2025_02_18",
#    "datalab/models/text_detection/2025_05_07",
#    "datalab/models/ocr_error_detection/2025_02_18",
#]
---

# parserPDF

[![Gradio](https://img.shields.io/badge/Gradio-SDK-amber?logo=gradio)](https://www.gradio.app/)
[![Python](https://img.shields.io/badge/Python->=3.12-blue?logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?logo=mit)](LICENSE)

A Gradio-based web application for converting PDF, HTML and Word documents to Markdown format. Powered by the Marker library (a pipeline of deep learning models for document parsing) and optional LLM integration for enhanced processing. Supports batch processing of files and directories via an intuitive UI.

## Features
- **PDF to Markdown**: Extract text, tables, and images from PDFs, HTMLs and Word documents using Marker.
- **HTML to Markdown**: Convert HTML files to clean Markdown. #Deprecated
- **Batch Processing**: Upload multiple files or entire directories.
- **LLM Integration**: Optional use of Hugging Face or OpenAI models for advanced conversion (e.g., via Llama or GPT models).
- **Customizable Settings**: Adjust model parameters, output formats (Markdown/HTML), page ranges, and more via the UI.
- **Output Management**: Generated Markdown files saved to a configurable output directory, with logs and download links.

## Project Structure
```
parserpdf/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point – launches the Gradio UI
├── pyproject.toml              # Project configuration
├── .env                        # Environment variables (e.g., API tokens)
├── .gitignore                  # Git ignore rules
├── converters/                 # Conversion logic
│   ├── __init__.py
│   ├── extraction_converter.py # Document extraction utilities
│   ├── pdf_to_md.py            # Marker-based PDF, HTML, Word → Markdown
│   ├── html_to_md.py           # HTML → Markdown  #Deprecated
│   └── md_to_pdf.py            # Markdown → PDF (pending full implementation)
├── file_handler/               # File handling utilities
│   ├── __init__.py
│   └── file_utils.py           # Helpers for files, directories, and paths
├── llm/                        # LLM client integrations
│   ├── __init__.py
│   ├── hf_client.py            # Hugging Face client wrapper  ##PutOnHold
│   ├── openai_client.py        # Marker OpenAI client         ##NotFullyImplemented
│   ├── llm_login.py            # Authentication handlers
│   └── provider_validator.py   # Provider validation
├── ui/                         # Gradio UI components
│   ├── __init__.py
│   └── gradio_ui.py            # UI layout, event handlers and coordination
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── config.py               # Configuration constants
│   ├── config.ini              # config file for settings
│   ├── logger.py               # Logging wrapper
│   ├── lib_loader.py           # loads weasyprint lib dependencies to environ
│   ├── get_config.py           # helper for getting configurations
│   ├── get_arg_name.py         # helper for getting argument names
│   └── utils.py                # General utilities and helpers
├── data/                       # Sample data and outputs (gitignored)
│   ├── output_dir/             # Output directory
│   ├── pdf/                    # Sample PDFs
├── logs/                       # Log files (gitignored)
├── tests/                      # Unit tests   ##ToBeUpdated
│   ├── tests_converter.py          # tests for converters
└── scrapyard/                  # Development scraps


[Projected]
├── transformers/
│   ├── __init__.py
│   ├── marker.py           # Marker class
│   └── marker_utils.py     # helpers for Marker class

```

## Installation
1. Clone the repository:
   ```
   git clone <repo-url>
   cd parserpdf
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional for LLM features):
   - Create a `.env` file with your API tokens, e.g.:
     ```
     HF_TOKEN=hf_xxx
     OPENAI_API_KEY=sk-xxx
     ```
   - HuggingFace login (oauth) integrated with Gradio:

4. Install Marker (if not in requirements.txt):
   ```
   pip install marker-pdf[full]
   ```

## Usage
1. Run the application:
   ```
   python main.py
   ```

2. Open the provided local URL (e.g., http://127.0.0.1:7860) in your browser.

3. In the UI:
   - Upload PDF/HTML/Word files or directories via the "PDF, HTML & Word ➜ Markdown" tab.
   - Configure LLM/Marker settings in the accordions (e.g., select provider, model, tokens).
   - Click "Process All Uploaded Files" to convert.
   - View logs, JSON output, and download generated Markdown files.

### Example Workflow
- Upload a PDF directory.
- Set model to `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (Hugging Face).
- Enable LLM if needed, set page range (e.g., "1-10").
- Process: Outputs Markdown files with extracted text/images to `output_dir`.

## Configuration
- Edit `utils/config.ini` or `utils/config.py` for defaults (e.g., model ID, output dir).
- On windows, set weasyprint's GTK path: e.g. "C:\\Dat\\dev\\gtk3-runtime\\bin" or "C:\\msys64\\mingw64\\bin"
- UI overrides: Adjust sliders for max tokens, temperature, workers, etc.

## LLM Providers
- **Hugging Face**: Supports inference providers like Fireworks AI, Together AI.
- **OpenAI**: Compatible via router (default: https://router.huggingface.co/v1).
- Login via UI or CLI: `huggingface-cli login`.

## Output
- Markdown files saved to `output_dir` (default: `./output_dir`).
- Images extracted as JPEGs alongside Markdown.
- Logs in `logs/` and UI textbox.

## Limitations & TODO
- Markdown → PDF is pending full implementation.
- HTML tab is deprecated; use main tab for mixed uploads.
- Large files/directories may require increased `max_workers` and higher processing power.
- No JSON/chunks output yet (flagged for future).

## Contributing
Fork the repo, create a branch, and submit a PR. 
- GitHub
- HuggingFace Space Community

Ensure tests pass: - verify the application's functionality. ##TardyOutdated
```
pytest tests/
```
Test Structure
- tests/test_converters.py: Tests PDF/HTML/Markdown converters, including init, conversion, batch processing, and error handling.
- tests/test_file_handler.py: Tests file collection utilities (PDF/HTML/MD paths), data processing, and output directory creation.
- tests/test_utils.py: Tests logging setup, config loading, utility functions like is_dict/is_list_of_dicts, and configuration access.
- tests/test_llm.py: Tests LLM login, provider validation, Hugging Face/OpenAI client initialization, and API interactions.
- tests/test_main_ui.py: Tests main application logic, UI building, batch conversion, file accumulation, and ProcessPoolExecutor integration.


## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Built with [Gradio](https://gradio.app/) for the UI.
- PDF parsing via [Marker](https://github.com/VikParuchuri/marker).
- LLM integrations using Hugging Face Transformers and OpenAI APIs.
- HuggingFace Spaces Configuration Reference [HF Spaces Configuration Reference](https://huggingface.co/docs/hub/en/spaces-config-reference)
- IBM Research: [HF Spaces Guide](https://huggingface.co/spaces/ibm-granite/granite-vision-demo/blob/main/DEVELOPMENT.md)