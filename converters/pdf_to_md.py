# converters/pdf_to_md.py
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import traceback  ## Extract, format and print information about Python stack traces.
import time

import spaces

from converters.extraction_converter import DocumentConverter  #, DocumentExtractor #as docextractor #ExtractionConverter  #get_extraction_converter  ## SMY: should disuse
from file_handler.file_utils import write_markdown, dump_images, collect_pdf_paths, collect_html_paths, collect_markdown_paths, create_outputdir

from utils import config
from utils.lib_loader import set_weasyprint_library
from utils.logger import get_logger

logger = get_logger(__name__)

# Define global variables   ##SMY: TODO: consider moving to Globals sigleton constructor
docconverter: DocumentConverter = None
converter = None  #DocumentConverter
#converter:DocumentConverter.converter = None

@spaces.GPU
# Define docextractor in the pool as serialised object and passed to each worker process.
# Note: DocumentConverter must be "picklable".
def init_worker(#self,
    provider: str,
    model_id: str,
    #base_url,
    hf_provider: str,
    endpoint_url: str,
    backend_choice: str,
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    api_token: str,
    openai_base_url: str,  #: str = "https://router.huggingface.co/v1",
    openai_image_format: str,  #: str | None = "webp",
    max_workers: int,
    max_retries: int,  #: int | None = 2,
    output_format: str,  #: str = "markdown",
    output_dir: str,  #: Union | None = "output_dir",
    use_llm: bool,  #: bool | None = False,
    force_ocr: bool,
    page_range: str,  #: str | None = None
    ):
    #'''
    """ 
    instantiate DocumentConverter/DocumentExtractor for use in each pool worker
    Args:

    """

    ## moved to class
    #    Initialise the global `converter` in each worker
    # Define global variables
    global docconverter
    global converter

    #'''
    # 1) Instantiate the DocumentConverter
    logger.log(level=20, msg="initialising docconverter:", extra={"model_id": model_id, "hf_provider": hf_provider})  ##debug

    try:
        docconverter = DocumentConverter(
            model_id,  #: str,
            hf_provider,  #: str,
            temperature,  #: float,
            top_p,  #: float,
            api_token,  #: str,
            openai_base_url,  #: str = "https://router.huggingface.co/v1",
            openai_image_format,  #: str | None = "webp",
            max_workers,  #: int  | None = 1,
            max_retries,  #: int | None = 2,
            output_format,  #: str = "markdown",
            output_dir,  #: Union | None = "output_dir",
            use_llm,  #: bool | None = False,
            force_ocr,
            page_range,  #: str | None = None
        )
        logger.log(level=20, msg="✔️ docextractor initialised:", extra={"docconverter model_id": docconverter.converter.config.get("openai_model"), "docconverter use_llm": docconverter.converter.use_llm, "docconverter output_dir": docconverter.output_dir})
    except Exception as exc:
        #logger.error(f"Failed to initialise DocumentConverter: {exc}")  #debug
        tb = traceback.format_exc()
        logger.exception(f"init_worker: Error initialising DocumentConverter → {exc}\n{tb}", exc_info=True)
        return f"✗ init_worker: error initialising DocumentConverter → {exc}\n{tb}"
    
    #docconverter = docconverter
    converter = docconverter.converter
    #self.llm_service = docconverter.llm_service  ##duplicate?
    #self.model_id = model_id   ##duplicate?
    #'''

class PdfToMarkdownConverter:
    """
    Wrapper around the Marker library that converts PDFs to Markdown.
    """

    #def __init__(self, options: Dict | None = None):
    def __init__(self, options: Dict | None = None): #extractor: DocumentExtractor, options: Dict | None = None):
        self.options = options or {}    ##SMY: TOBE implemented - bring all Marker's options
        self.output_dir_string = ''
        self.output_dir = self.output_dir_string  ## placeholder
        #self.OUTPUT_DIR = config.OUTPUT_DIR     ##flag unused
        #self.MAX_RETRIES = config.MAX_RETRIES   ##flag unused
        #self.docconverter = None  #DocumentConverter
        #self.converter = self.docconverter.converter #None 

    # This global will be set (re-initialised) in each worker after init_worker runs

    
    #duration = 5.75 * pdf_files_count if pdf_files_count>=2 else 7
    #duration = 10
    #@spaces.GPU(duration=duration)   ## HF Spaces GPU support
    @spaces.GPU
    ## moved from extraction_converter ( to standalone extract_to_md)
    #def extract(self, src_path: str, output_dir: str) -> Dict[str, int, Union[str, Path]]:
    def extract(self, src_path: str, output_dir: str):   #Dict:
    #def extract(src_path: str, output_dir: str) -> Dict[str, int]:  #, extractor: DocumentExtractor) -> Dict[str, int]:
        """
        Convert one file (PDF/HTML) to Markdown + images.
        Writes a `.md` file and any extracted images under `output_dir`.
        Returns a dict with metadata, e.g. {"filename": <file.name>, "images": <count>, "filepath": <filepath>}.
        """
        
        from globals import config_load_models
        try:
            ## SMY: TODO: convert htmls to PDF. Marker will by default attempt weasyprint which typically raise 'libgobject-2' error on Win
            weasyprint_libpath = config_load_models.weasyprint_libpath if config_load_models.weasyprint_libpath else None
            # Set a new environment variable
            set_weasyprint_library(weasyprint_libpath)  ##utils.lib_loader.set_weasyprint_library()
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error loading weasyprint backend dependency → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            raise RuntimeWarning(f"✗ error during loading weasyprint backend dependency → {exc}\n{tb}")

        
        # Run Marker conversion with LLM if use_llm is true
        try:
            #rendered = self.docconverter.converter(src_path, use_llm=True)
            #rendered = self.docconverter.converter(src_path)
            rendered = converter(src_path)
            logger.log(level=20, msg=f"✓ File extraction successful for {Path(src_path).name}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error during file extraction → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            
            return f"✗ error during extraction → {exc}\n{tb}"

        # Write Markdown file
        '''
        base = Path(str_path).stem   ## Get filename without extension
        md_path = output_dir / f"{base}.md"  # Join output dir and new markdown file with the slash operator
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(rendered.markdown)
        '''
        try:
            md_file = write_markdown(src_path=src_path, output_dir=output_dir, rendered=rendered)
            #debug md_file = "debug_md_file dummy name" ##debug
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"✗ error creating md_file → {exc}\n{tb}", exc_info=True)
            #return f"✗ error creating md_file → {exc}\n{tb}"        

        # Dump extracted images
        #debug images_count = 100  ##debug
        try:
            images_count, image_path = dump_images(src_path, output_dir, rendered)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"✗ error counting and creating image_path → {exc}\n{tb}", exc_info=True)
            #return f"✗ error counting andcreating image_path → {exc}\n{tb}"
        
        #return {"images": len(rendered.images), "file": md_file}  ##debug
        return {"file": md_file.name, "images": images_count, "filepath": md_file, "image_path": image_path}  ####SMY should be Dict[str, int, str]. Dicts are not necessarily ordered.

    #def convert_files(src_path: str, output_dir: str, max_retries: int = 2) -> str:
    #def convert_files(self, src_path: str, output_dir_string: str = None, max_retries: int = 2, progress = gr.Progress()) -> Union[Dict, str]:  #str:    
    def convert_files(self, src_path: str, max_retries: int = 2) -> Union[Dict, str]:
    #def convert_files(self, src_path: str) -> str:    
        """
        Worker task: use `extractor` to convert file with retry/backoff.
        Returns a short log line.
        """

        '''try:   ##moved to gradio_ui. sets to PdfToMarkdownConverter.output_dir_string
            output_dir = create_outputdir(root=src_path, output_dir_string=self.output_dir_string)
            logger.info(f"✓ output_dir created: {output_dir}")  #{create_outputdir(src_path)}"            
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("✗ error creating output_dir → {exc}\n{tb}", exc_info=True)
            return f"✗ error creating output_dir → {exc}\n{tb}"'''
        output_dir = Path(self.output_dir)  ## takes the value from gradio_ui

        try:
            #if Path(src_path).suffix.lower() not in {".pdf", ".html", ".htm"}:
            #if not Path(src_path).name.endswith(tuple({".pdf", ".html"})):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):
            #if not Path(src_path).name.endswith((".pdf", ".html", ".docx", ".doc")):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):            
            if not Path(src_path).name.endswith(config.file_types_tuple):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):
                logger.log(level=20, msg=f"skipped {Path(src_path).name}", exc_info=True)
                return f"skipped {Path(src_path).name}"
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("✗ error during suffix extraction → {exc}\n{tb}", exc_info=True)
            return f"✗ error during suffix extraction → {exc}"

        #max_retries = self.MAX_RETRIES
        for attempt in range(1, max_retries + 1):
            try:
                #info = self.extract(str(src_path), str(output_dir.stem))  #extractor.converter(str(src_path), str(output_dir))  #
                info = self.extract(str(src_path), str(output_dir))  #extractor.converter(str(src_path), str(output_dir))  #
                logger.log(level=20, msg=f"✓ : info about extracted {Path(src_path).name}: ", extra={"info": str(info)})
                '''  ##SMY: moving formating to calling Gradio
                img_count = info.get("images", 0)
                md_filename = info.get("file", 0)
                md_filepath = info.get("filepath", 0)
                #return f"✓ {src_path.name} ({img_count} images)"
                return f"✓ {md_filename}: ({img_count} images)", md_filepath
                '''
                return info  ##SMY: simply return the dict
            except Exception as exc:
                if attempt == max_retries:
                    tb = traceback.format_exc()
                    return f"✗ {info.get('file', 'UnboundlocalError: info is None')} → {exc}\n{tb}"
                    #return f"✗ {md_filename} → {exc}\n{tb}"
                
                #time.sleep(2 ** attempt)
                # Exponential backoff before retry
                logger.warning(f"Attempt {attempt} failed for {Path(src_path).name}: {exc}. Retrying in {2 ** attempt}s...")

                time.sleep(2 ** attempt)
    
    ## SMY: unused
    #=====================  discarded
    '''
    def convert(self, pdf_path: Path) -> str:
        """
        Convert a single PDF file to Markdown string.

        Parameters
        ----------
        pdf_path : pathlib.Path
            Path to the source PDF.

        Returns
        -------
        str
            The extracted Markdown content.
        """
        logger.info(f"Converting {pdf_path} → Markdown")
        try:
            md_text = self.marker.extract_markdown(str(pdf_path))
            return md_text
        except Exception as exc:
            logger.exception("Marker failed to convert PDF.")
            raise RuntimeError(f"Failed to convert {pdf_path}") from exc

    
    def batch_convert(self, pdf_paths: List[Path]) -> Dict[str, str]:
        """
        Convert multiple PDFs and return a mapping of filename → Markdown.

        Parameters
        ----------
        pdf_paths : list[pathlib.Path]
            List of PDF files to process.

        Returns
        -------
        dict
            Mapping from original file name (without extension) to Markdown string.
        """
        results = {}
        for p in pdf_paths:
            try:
                md = self.convert(p)
                key = p.stem  # filename without .pdf
                results[key] = md
            except Exception as exc:
                logger.warning(f"Skipping {p}: {exc}")
        return results

    def convert_file(self, src_path: Path, extractor: DocumentConverter): #DocumentExtractor):  #-> str:
        """
        Converts one PDF or HTML file to Markdown + images
        with retry/backoff on errors.
        """
        path    = src_path
        out_dir = path.parent / self.OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                rendered = extractor.converter(str(path), use_llm=True)

                # Write Markdown
                md_file = out_dir / f"{path.stem}.md"
                md_file.write_text(rendered.markdown, encoding="utf-8")

                # Dump images
                for name, content in rendered.images.items():
                    (out_dir / name).write_bytes(content)

                print(f"[ok]   {path.name}")
                return

            except Exception as e:
                if attempt == self.MAX_RETRIES:
                    print(f"[fail] {path.name} after {attempt} attempts")
                    traceback.print_exc()
                else:
                    backoff = 2 ** attempt
                    print(f"[retry] {path.name} in {backoff}s ({e})")
                    time.sleep(backoff)
    '''