yaml
---
title: "parser2md" - PDF & HTML parser to markdown
sdk: gradio
command: python main.py
---

[project]
---
name: "parserpdf" - PDF & HTML parser to markdown
#title: "parserPDF"
title: "parser2md"
sdk: gradio
#sdk_version: 5.0.1
command: python main.py
app_file: main.py
emoji: 📝
colorFrom: yellow
colorTo: purple
name: "parser2md"
pinned: false
license: mit
short_description: 'PDF & HTML parser to markdown'
version: "0.1.0"
readme: "README.md"
requires-python: ">=3.12"
dependencies: []
owner: "research-semmyk"
---

# parserPDF

[![Gradio](https://img.shields.io/badge/Gradio-SDK-amber?logo=gradio)](https://www.gradio.app/)
[![Python](https://img.shields.io/badge/Python->=3.12-blue?logo=python)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?logo=mit)](LICENSE)

A Gradio-based web application for converting PDF and HTML documents to Markdown format. Powered by the Marker library (a pipeline of deep learning models for document parsing) and optional LLM integration for enhanced processing. Supports batch processing of files and directories via an intuitive UI.

## Features
- **PDF to Markdown**: Extract text, tables, and images from PDFs using Marker.
- **HTML to Markdown**: Convert HTML files to clean Markdown.
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
│   ├── pdf_to_md.py            # Marker-based PDF → Markdown
│   ├── html_to_md.py           # HTML → Markdown
│   └── md_to_pdf.py            # Markdown → PDF (pending full implementation)
├── file_handler/               # File handling utilities
│   ├── __init__.py
│   └── file_utils.py           # Helpers for files, directories, and paths
├── llm/                        # LLM client integrations
│   ├── __init__.py
│   ├── hf_client.py            # Hugging Face client wrapper
│   ├── openai_client.py        # Marker OpenAI client
│   ├── llm_login.py            # Authentication handlers
│   └── provider_validator.py   # Provider validation
├── ui/                         # Gradio UI components
│   ├── __init__.py
│   └── gradio_ui.py            # UI layout and event handlers
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
├── tests/                      # Unit tests
├── tests_converter.py          # tests for converters
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

4. Install Marker (if not in requirements.txt):
   ```
   pip install marker-pdf
   ```

## Usage
1. Run the application:
   ```
   python main.py
   ```

2. Open the provided local URL (e.g., http://127.0.0.1:7860) in your browser.

3. In the UI:
   - Upload PDF/HTML files or directories via the "PDF & HTML ➜ Markdown" tab.
   - Configure LLM/Marker settings in the accordions (e.g., select provider, model, tokens).
   - Click "Process All Uploaded Files" to convert.
   - View logs, JSON output, and download generated Markdown files.

### Example Workflow
- Upload a PDF directory.
- Set model to `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (Hugging Face).
- Enable LLM if needed, set page range (e.g., "1-10").
- Process: Outputs Markdown files with extracted text/images to `output_dir`.

## Configuration
- Edit `utils/config.py` or `utils/config.ini` for defaults (e.g., model ID, output dir).
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
- Large files/directories may require increased `max_workers`.
- No JSON/chunks output yet (flagged for future).

## Contributing
Fork the repo, create a branch, and submit a PR. 

Ensure tests pass: - verify the application's functionality.
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