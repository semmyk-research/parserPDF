# converters/pdf_to_md.py
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import traceback  ## Extract, format and print information about Python stack traces.
import time

from gradio import Progress as grP
import spaces
from globals import config_load_models, config_load

from converters.extraction_converter import DocumentConverter  #, DocumentExtractor #as docextractor #ExtractionConverter  #get_extraction_converter  ## SMY: should disuse
from utils.file_utils import write_markdown, dump_images, collect_pdf_paths, collect_html_paths, collect_markdown_paths, create_outputdir

#from utils import config
from utils.lib_loader import set_weasyprint_library
from utils.logger import get_logger

logger = get_logger(__name__)

# Define global variables   ##SMY: TODO: consider moving to Globals sigleton constructor
## moved to class
#docconverter: DocumentConverter = None
#converter = None  #DocumentConverter
# Define docextractor in the pool as serialised object and passed to each worker process.
# Note: DocumentConverter must be "picklable".

#def init_worker(#self, ...

class PdfToMarkdownConverter:
    """
    Wrapper around the Marker library that converts PDFs to Markdown.
    """

    #def __init__(self, options: Dict | None = None):
    def __init__(self, options: Dict | None = None): #extractor: DocumentExtractor, options: Dict | None = None):
        self.options = options or {}    ##SMY: TOBE implemented - bring all Marker's options
        self.output_dir_string = ''
        self.output_dir = ''   #self.output_dir_string  ## placeholder
        self.docconverter = None  #DocumentConverter
        self.converter = None  #self.docconverter.converter #None 
    
    def init_docconverter(self, output_dir: Union[str, Path] = config_load.output_dir, progress3=grP(track_tqdm=True)):
        #'''
        """ 
        instantiate DocumentConverter/DocumentExtractor for use
        Args:
            ##TODO
        """
        
        provider: str = config_load.provider
        model_id: str = config_load.model_id
        #base_url,
        hf_provider: str = config_load.hf_provider
        endpoint_url: str = config_load.endpoint
        backend_choice: str = config_load.backend_choice
        system_message: str = config_load.system_message
        max_tokens: int = config_load.max_tokens
        temperature: float = config_load.temperature
        top_p: float = config_load.top_p
        stream: bool = config_load.stream
        api_token: str = config_load.api_token
        openai_base_url: str = config_load.openai_base_url
        openai_image_format: str = config_load.openai_image_format
        max_workers: int = config_load.max_workers
        max_retries: int = config_load.max_retries
        debug: bool = config_load.debug
        output_format: str = config_load.output_format
        output_dir: Union[str, Path] = config_load.output_dir_string   #output_dir #
        use_llm: bool = config_load.use_llm
        force_ocr: bool = config_load.force_ocr
        strip_existing_ocr: bool = config_load.strip_existing_ocr
        disable_ocr_math: bool = config_load.disable_ocr_math
        page_range: str = config_load.page_range
        

        # 1) Instantiate the DocumentConverter
        logger.log(level=20, msg="initialising docconverter:", extra={"model_id": model_id, "hf_provider": hf_provider})  ##debug
        progress3((0,1), desc=f"initialising docconverter: ...")
        #progress2((10,16), desc=f"ProcessPoolExecutor: Pooling file conversion result: [{str(result_interim)}[:20]]")
        time.sleep(0.75)  #.sleep(0.25)

        try:
            docconverter = DocumentConverter(
                model_id,   #: str,
                hf_provider,    #: str,
                temperature,    #: float,
                top_p,          #: float,
                api_token,  #: str,
                openai_base_url,    #: str = "https://router.huggingface.co/v1",
                openai_image_format,    #: str | None = "webp",
                max_workers,      #: int  | None = 1,
                max_retries,      #: int | None = 2,
                debug,                  #: bool = False
                output_format,  #: str = "markdown",
                output_dir,        #: Union | None = "output_dir",
                use_llm,              #: bool | None = False,
                force_ocr,          #: bool | None = False,
                strip_existing_ocr, #bool = False,
                disable_ocr_math,     #bool = False,
                page_range,        #: str | None = None
            )
            logger.log(level=20, msg="✔️ docextractor initialised:", extra={"docconverter model_id": docconverter.converter.config.get("openai_model"), "docconverter use_llm": docconverter.converter.use_llm, "docconverter output_dir": docconverter.output_dir})
            progress3((1,1), desc=f"✔️ docextractor initialised:")
            time.sleep(0.75)  #.sleep(0.25)
        except Exception as exc:
            #logger.error(f"Failed to initialise DocumentConverter: {exc}")  #debug
            tb = traceback.format_exc()
            logger.exception(f"init_worker: Error initialising DocumentConverter → {exc}\n{tb}", exc_info=True)
            return f"✗ init_worker: error initialising DocumentConverter → {exc}\n{tb}"
        
        converter = docconverter.converter
        self.docconverter = docconverter
        self.converter = converter

        #return converter
    
    #duration = 60*config_load_models.pdf_files_count if config_load_models.pdf_files_count>=10 else 360  ## sec
    duration = 60*config_load_models.pdf_files_count if config_load_models.use_llm else 90  ## sec
    @spaces.GPU(duration=duration)   ## HF Spaces GPU support
    def extract(self, src_path: str, output_dir: str):   ##-> Dict[str, int, Union[str, Path]]:
    #def extract(self, src_path: str, output_dir: str, progress4=grP()):   #Dict:
    ###def extract(src_path: str, output_dir: str) -> Dict[str, int]:  #, extractor: DocumentExtractor) -> Dict[str, int]:
        """
        Convert one file (PDF/HTML) to Markdown + images.
        Writes a `.md` file and any extracted images under `output_dir`.
        Returns a dict with metadata, e.g. {"filename": <file.name>, "images": <count>, "filepath": <filepath>}.
        """
        
        #from globals import config_load_models   ##SMY: moved to top-level import
        try:
            ## SMY: TODO: convert htmls to PDF. Marker will by default attempt weasyprint which typically raise 'libgobject-2' error on Win
            weasyprint_libpath = config_load_models.weasyprint_libpath if config_load_models.weasyprint_libpath else None
            # Set a new environment variable
            set_weasyprint_library(weasyprint_libpath)  ##utils.lib_loader.set_weasyprint_library()
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error loading weasyprint backend dependency → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            raise RuntimeWarning(f"✗ error during loading weasyprint backend dependency → {exc}\n{tb}")

        # Initialise Marker Converter
        try:
            if not self.converter:
                self.init_docconverter(output_dir)

            logger.log(level=20, msg=f"✓ Initialised Marker Converter")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error during Marker Converter initialisation → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            
            return f"✗ error during extraction → {exc}\n{tb}"
        
        # Run Marker conversion with LLM if use_llm is true
        try:
            #progress4((0,1), desc=f"Extracting File: {Path(src_path).name}")
            #time.sleep(0.75)  #.sleep(0.25)
            
            #rendered = self.docconverter.converter(src_path)
            rendered = self.converter(src_path)

            logger.log(level=20, msg=f"✓ File extraction successful for {Path(src_path).name}")
            #progress4((1,1), desc=f"✓ File extraction successful for {Path(src_path).name}")
            #time.sleep(0.75)  #.sleep(0.25)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error during file extraction → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            
            return f"✗ error during extraction → {exc}\n{tb}"

        # Write Markdown file
        try:
            md_file = write_markdown(src_path=src_path, output_dir=output_dir, rendered=rendered, output_format=config_load.output_format)
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

    #duration = 60*config_load_models.pdf_files_count if config_load_models.pdf_files_count>=10 else 360  ## sec
    #@spaces.GPU(duration=duration)   ## HF Spaces GPU support
    
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
        #output_dir = Path(self.output_dir)  ## takes the value from gradio_ui
        output_dir = Path(config_load.output_dir)  # Takes the value when output_dir is created in gradio_process
        self.output_dir = output_dir

        try:
            #if Path(src_path).suffix.lower() not in {".pdf", ".html", ".htm"}:
            #if not Path(src_path).name.endswith(tuple({".pdf", ".html"})):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):
            #if not Path(src_path).name.endswith((".pdf", ".html", ".docx", ".doc")):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):            
            if not Path(src_path).name.endswith(config_load.file_types_tuple):  #,".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"})):
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