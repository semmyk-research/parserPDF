# utils/config.py

import os

"""
Centralised configuration constants.
##SMY: TODO: Create Class Settings(BaseSettings)  leveraging from pydantic_settings import BaseSettings
"""

# UI text
TITLE = "Parser2md (PyPDFmd/ParserPDF) – PDF & HTML ↔ Markdown Converter"
DESCRIPTION = (
    "Parser2md (PyPDFmd) - Convert files to Markdown."
)
DESCRIPTION_PDF_HTML = (
    "Upload a single or multiple PDF or HTML, a folder or an entire directory tree "
    "to convert to Markdown."
)
DESCRIPTION_PDF = (
    "Drag‑and‑drop a single PDF, a folder of PDFs or an entire directory tree "
    "to convert to Markdown."
)
DESCRIPTION_HTML = (
    "Drag‑and‑drop a single HTML, a folder of HTMLs or an entire directory tree "
    "to convert to Markdown."
)
DESCRIPTION_MD = (
    "Upload Markdown/LaTeX files and generate a polished PDF."
)

# File types
file_types_list  = []
file_types_tuple = (".pdf", ".html", ".docx", ".doc")
#file_types_list = list[file_types_tuple]
#file_types_list.extend(file_types_tuple)


# Conversion defaults
DEFAULT_MARKER_OPTIONS = {
    "include_images": True,
    "image_format": "png",
}

# Configuration
MAX_WORKERS    = int(os.getenv("MAX_WORKERS", "4"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "2"))  #3
INPUT_DIR      = os.getenv("INPUT_DIR",  "inputs")  # unused
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "md_output")
USE_LLM             = bool(os.getenv("USE-LLM", False))  #True
EXTRACT_IMAGES      = bool(os.getenv("EXTRACT_IMAGES", True))  #True
OUTPUT_IMAGE_FORMAT = os.getenv("OUTPUT_IMAGE_FORMAT", "png")  #png
OUTPUT_ENCODING     = os.getenv("OUTPUT_ENCODING", "utf-8")  #utf-8
DEBUG_DATA_FOLDER   = os.getenv("DEBUG_DATA_FOLDER", "debug_data")  #debug_data

# Global
HF_MODEL       = os.getenv("HF_MODEL", "gpt2")  # swap for a chat-capable model
HF_TOKEN       = os.getenv("HF_TOKEN")          # your Hugging Face token



## //TODO: 
# from config.ini  ##SMY: future plan to merge
api_token="a1b2c3"
OUTPUT_FORMAT       = "markdown"   #output_format
OPENAI_MODEL        = "openai/gpt-oss-120b"   #openai_model
OPENAI_API_KEY      = "a1b2c3"  #openai_api_key
OPENAI_BASE_URL     = "https://router.huggingface.co/v1"   ##openai_base_url
OPENAI_IMAGE_FORMAT = "webp"    #openai_image_format
OUTPUT_IMAGE_FORMAT = "png"
#max_retries=3 

#[marker]
PROVIDER       = "openai"  #provider
MODEL_ID       = "openai/gpt-oss-120b"  #model_id
HF_PROVIDER    = "fireworks-ai"  #hf_provider
ENDPOINT_URL   = ""  #endpoint_url
BACKEND_CHOiCE = "provider"  #backend_choice
SYSTEM_MESSAGE = ""  #system_message
MAX_TOKENS     = 8192  #max_tokens
TEMMPERATURE   = 0.2  #temperature
TOP_P          = 0.2  #top_p
STREAM         = True  #stream

# Globals within each worker process
hf_client      = None
artifact_dict  = None
pdf_converter  = None
html_converter = None

