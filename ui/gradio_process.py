# ui/gradio_process.py

from re import Match
from unittest import result
import gradio as gr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

import time

from pathlib import Path, WindowsPath
from typing import Optional, Union, Literal #, Dict, List, Any, Tuple

from huggingface_hub import get_token
import spaces    ##HuggingFace spaces to accelerate GPU support on HF Spaces

#import utilities, helpers
#import utils.file_utils
from utils.file_utils import zip_processed_files, process_dicts_data, create_temp_folder   #, collect_pdf_paths, collect_html_paths, collect_markdown_paths, create_outputdir  ## should move to handling file
from utils.config import TITLE, DESCRIPTION, DESCRIPTION_PDF_HTML, DESCRIPTION_PDF, DESCRIPTION_HTML, DESCRIPTION_MD  #, file_types_list, file_types_tuple
from utils.utils import is_dict, is_list_of_dicts
from utils.get_config import get_config_value

from llm.llm_login import get_login_token, is_loggedin_huggingface, login_huggingface
from converters.extraction_converter import DocumentConverter as docconverter  #DocumentExtractor #as docextractor
from converters.pdf_to_md import PdfToMarkdownConverter   #, init_worker
#from converters.md_to_pdf import MarkdownToPdfConverter  ##SMY: PENDING: implementation

import traceback  ## Extract, format and print information about Python stack traces.
from utils.logger import get_logger

logger = get_logger(__name__)   ##NB: setup_logging()  ## set logging

# Instantiate converters class once – they are stateless
pdf2md_converter = PdfToMarkdownConverter()
#md2pdf_converter = MarkdownToPdfConverter()

    
# User eXperience: Load Marker models ahead of time if not already loaded in reload mode
## SMY: 29Sept2025 - Came across https://github.com/xiaoyao9184/docker-marker/tree/master/gradio
from converters.extraction_converter import load_models
from globals import config_load_models
try:
    if not config_load_models.model_dict:
        model_dict = load_models()
        config_load_models.model_dict = model_dict
    '''if 'model_dict' not in globals():
        global model_dict
        model_dict = load_models()'''
    logger.log(level=30, msg="Config_load_model: ", extra={"model_dict": str(model_dict)})
except Exception as exc:
    #tb = traceback.format_exc()   #exc.__traceback__
    logger.exception(f"✗ Error loading models (reload): {exc}")  #\n{tb}")
    raise RuntimeError(f"✗ Error loading models (reload): {exc}")  #\n{tb}") 

#def get_login_token( api_token_arg, oauth_token: gr.OAuthToken | None=None,):  ##moved to llm_login


#duration = 5.75 * pdf_files_count if pdf_files_count>=2 else 7
#@spaces.GPU(duration=duration)   ## HF Spaces GPU support
def get_results_files_conversion(pdf_files, pdf_files_count, progress2=gr.Progress(track_tqdm=True)):
    #Use progress.tqdm to integrate with the executor map

    results = []
    
    #for result_interim in progress2.tqdm(
    for i, pdf_file in enumerate(iterable=progress2.tqdm(
                iterable=pdf_files,  #, max_retries), total=len(pdf_files)
                desc=f"Processing file conversion ... pool.map",
                total=pdf_files_count)
                ):
        result_interim = pdf2md_converter.convert_files(pdf_file)

        # Update the Gradio UI to improve user-friendly eXperience
        #yield gr.update(interactive=True), f"ProcessPoolExecutor: Pooling file conversion result: [{str(result_interim)}[:20]]", {"process": "Processing files ..."}, f"dummy_log.log"
        progress2((i,pdf_files_count), desc=f"Processing file conversion result: {i}: {str(pdf_file)} : [{str(result_interim)[:20]}]")
        #progress2((10,16), desc=f"ProcessPoolExecutor: Pooling file conversion result: [{str(result_interim)}[:20]]")
        time.sleep(0.75)  #.sleep(0.25)
        
        results.append(result_interim)
        
    return results

def get_results_files_conversion_with_pool(pdf_files, pdf_files_count, max_workers: int, progress2=gr.Progress(track_tqdm=True)):
    #Use progress.tqdm to integrate with the executor map

    results = []
    try:
        # Create a pool with init_worker initialiser
        ##SMY: dropped ProcessPoolExecutor due to slow Marker conversion.Marker already leverage ThreadPoolExecutor and ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=max_workers,
            ) as pool:
    
                logger.log(level=30, msg="Initialising ProcessPoolExecutor: pool:", extra={"pdf_files": pdf_files[:3], "files_len": len(pdf_files), "progress": str(progress2),}) 
                progress2((10,16), desc=f"Starting ProcessPool queue: Processing Files ...")
                time.sleep(0.25)

                # Map the files (pdf_files) to the conversion function (pdf2md_converter.convert_file)
                #try:
                    #yield gr.update(interactive=True), f"ProcessPoolExecutor: Pooling file conversion ...", {"process": "Processing files ..."}, f"dummy_log.log"
                #    progress((9,16), desc=f"ProcessPoolExecutor: Pooling file conversion ...")
                #    time.sleep(0.25)
                #    yield gr.update(interactive=False), f"ProcessPoolExecutor: Pooling file conversion ...", {"process": "Processing files ..."}, f"dummy_log.log"
    
                # Use progress.tqdm to integrate with the executor mapresults = pool.map(pdf2md_converter.convert_files, pdf_files)  ##SMY iterables  #max_retries #output_dir_string)
                for i, result_interim in enumerate(progress2.tqdm(
                    iterable=pool.map(pdf2md_converter.convert_files, pdf_files),  #, max_retries), total=len(pdf_files)
                    desc="ProcessPoolExecutor: Pooling file conversion ...",
                    total=pdf_files_count, unit="files")
                    ):

                        results.append(result_interim)
        
                        # Update the Gradio UI to improve user-friendly eXperience
                        yield gr.update(interactive=True), f"ProcessPoolExecutor: Pooling file conversion result: {i} : [{str(result_interim)[:20]}]", {"process": "Processing files ..."}, f"dummy_log.log"
                        #progress((10,16), desc=f"ProcessPoolExecutor: Pooling file conversion result: [{str(result_interim)[:20]}]")
                        progress2((i, pdf_files_count), desc=f"ProcessPoolExecutor: Pooling file conversion result: {i} : [{str(result_interim)[:20]}]")
                        time.sleep(0.25)
    except Exception as exc:
        # Raise the exception to stop the Gradio app: exception to halt execution
        logger.exception("Error during pooling file conversion", exc_info=True)  # Log the full traceback
        tbp = traceback.print_exc()  # Print the exception traceback
        # Update the Gradio UI to improve user-friendly eXperience
        yield gr.update(interactive=True), f"An error occurred during pool.map: {str(exc)}", {"Error":f"Error: {exc}\n{tbp}"}, f"dummy_log.log"  ## return the exception message
        return [gr.update(interactive=True), f"An error occurred during pool.map: {str(exc)}", {"Error":f"Error: {exc}\n{tbp}"}, f"dummy_log.log"]  ## return the exception message
        ##======
        
    return results

def get_results_files_conversion_with_pool_ascomplete(pdf_files, pdf_files_count, max_workers: int, progress2=gr.Progress(track_tqdm=True)):
    """
        This function wraps the as_completed call to process results
        as they become available.
    """
    #Use progress.tqdm to integrate with the executor map

    results = []
    try:
        # Create a pool with init_worker initialiser
        ##SMY: dropped ProcessPoolExecutor due to slow Marker conversion.Marker already leverage ThreadPoolExecutor and ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=max_workers,
            ) as pool:
    
                logger.log(level=30, msg="Initialising ProcessPoolExecutor: pool:", extra={"pdf_files": pdf_files, "files_len": len(pdf_files), "progress": str(progress2)})  #pdf_files_count
                progress2((10,16), desc=f"Starting ProcessPool queue: Processing Files ...")
                time.sleep(0.25)

                # Submit each task individually and collect the futures
                futures = [pool.submit(pdf2md_converter.convert_files, file) for file in pdf_files]
                
                # Use progress.tqdm to integrate with the executor mapresults = pool.map(pdf2md_converter.convert_files, pdf_files)  ##SMY iterables  #max_retries #output_dir_string)
                for i, future in enumerate(progress2.tqdm(
                    iterable=as_completed(futures),  #pdf_files,
                    desc="ProcessPoolExecutor: Pooling file conversion ...",
                    total=pdf_files_count, unit="files")
                    ):
                        result_interim = future.result()
                        results.append(result_interim)
        
                        # Update the Gradio UI to improve user-friendly eXperience
                        yield gr.update(interactive=True), f"ProcessPoolExecutor: Pooling file conversion result: {i} : [{str(result_interim)[:20]}]", {"process": "Processing files ..."}, f"dummy_log.log"
                        #progress((10,16), desc=f"ProcessPoolExecutor: Pooling file conversion result: [{str(result_interim)[:20]}]")
                        progress2((i, pdf_files_count), desc=f"ProcessPoolExecutor: Pooling file conversion result: {i} : [{str(result_interim)[:20]}]")
                        time.sleep(0.25)
    except Exception as exc:
        # Raise the exception to stop the Gradio app: exception to halt execution
        logger.exception("Error during pooling file conversion", exc_info=True)  # Log the full traceback
        tbp = traceback.print_exc()  # Print the exception traceback
        # Update the Gradio UI to improve user-friendly eXperience
        yield gr.update(interactive=True), f"An error occurred during pool.map: {str(exc)}", {"Error":f"Error: {exc}\n{tbp}"}, f"dummy_log.log"  ## return the exception message
        return [gr.update(interactive=True), f"An error occurred during pool.map: {str(exc)}", {"Error":f"Error: {exc}\n{tbp}"}, f"dummy_log.log"]  ## return the exception message
        ##======
        
    return results

##SMY: TODO: future: refactor to gradio_process.py and 
## pull options to cli-options{"output_format":, "output_dir_string":, "use_llm":, "page_range":, "force_ocr":, "debug":, "strip_existing_ocr":, "disable_ocr_math""}
#@spaces.GPU
def convert_batch(
    pdf_files, #: list[str],
    pdf_files_count: int,
    provider: str,
    model_id: str,
    #base_url: str
    hf_provider: str,
    endpoint: str,
    backend_choice: str,
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    api_token_gr: str,
    #max_workers: int,
    #max_retries: int,
    openai_base_url: str = "https://router.huggingface.co/v1",
    openai_image_format: Optional[str] = "webp",
    max_workers: Optional[int] = 1,  #4,
    max_retries: Optional[int] = 2,
    debug: bool = False,        #Optional[bool] = False,  #True,
    #output_format: str = "markdown",
    output_format: Literal["markdown", "json", "html"] = "markdown",
    #output_dir: Optional[Union[str, Path]] = "output_dir",
    output_dir_string: str = "output_dir_default",
    use_llm: bool = False,      #Optional[bool] = False,  #True,
    force_ocr: bool = True,     #Optional[bool] = False,
    strip_existing_ocr: Optional[bool] = None,  #bool = False,
    disable_ocr_math: Optional[bool] = None,    #bool = False,
    page_range: str = None,     #Optional[str] = None,
    weasyprint_dll_directories: str = None,     #weasyprint_libpath 
    tz_hours: str = None,
    pooling: str = "no_pooling",   #bool = True,
    oauth_token: gr.OAuthToken | None=None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  #Progress tracker to keep tab on pool queue executor
    progress1: gr.Progress = gr.Progress(),
    #progress2: gr.Progress = gr.Progress(track_tqdm=True),
    ): #-> str:
    """
    Handles the conversion process using multiprocessing.
    Spins up a pool and converts all uploaded files in parallel.
    Aggregates per-file logs into one string.
    Receives Gradio component values, starting with the list of uploaded file paths
    """

    # login: Update the Gradio UI to improve user-friendly eXperience - commencing
    # [template]: #outputs=[process_button, log_output, files_individual_JSON, files_individual_downloads],            
    yield gr.update(interactive=False), f"Commencing Processing ... Getting login", {"process": "Commencing Processing"}, f"dummy_log.log"
    progress((0,16), f"Commencing Processing ...")
    time.sleep(0.25)
    
    # get token from logged-in user: 
    api_token = get_login_token(api_token_arg=api_token_gr, oauth_token=oauth_token)
    ##SMY: Strictly debug. Must not be live
    #logger.log(level=30, msg="Commencing: get_login_token", extra={"api_token": api_token, "api_token_gr": api_token_gr})

    '''try:
        ##SMY: might deprecate. To replace with oauth login from Gradio ui or integrate cleanly.
        #login_huggingface(api_token)  ## attempt login if not already logged in. NB: HF CLI login prompt would not display in Process Worker.
        
        if is_loggedin_huggingface() and (api_token is None or api_token == ""):
            api_token = get_token()   ##SMY: might be redundant
        
        elif is_loggedin_huggingface() is False and api_token:
            login_huggingface(api_token)
            # login: Update the Gradio UI to improve user-friendly eXperience
            #yield gr.update(interactive=False), f"login to HF: Processing files...", {"process": "Processing files"}, f"dummy_log.log"
        else:
            pass
            # login: Update the Gradio UI to improve user-friendly eXperience
            #yield gr.update(interactive=False), f"Not logged in to HF: Processing files...", {"process": "Processing files"}, f"dummy_log.log"
        
    except Exception as exc:  # Catch all exceptions
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during login_huggingface → {exc}\n{tb}", exc_info=True) # Log the full traceback
        return [gr.update(interactive=True), f"✗ An error occurred during login_huggingface → {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"]  # return the exception message
    '''
    progress((1,16), desc=f"Log in: {is_loggedin_huggingface(api_token)}")
    time.sleep(0.25)
    ## debug
    #logger.log(level=30, msg="pdf_files_inputs", extra={"input_arg[0]:": pdf_files[0]})

    #if not files:
    if not pdf_files or pdf_files is None:  ## Check if files is None. This handles the case where no files are uploaded.
        logger.log(level=30, msg="Initialising ProcessPool: No files uploaded.", extra={"pdf_files": pdf_files, "files_len": pdf_files_count})
        #outputs=[log_output, files_individual_JSON, files_individual_downloads],
        return [gr.update(interactive=True), "Initialising ProcessPool: No files uploaded.", {"Upload":"No files uploaded"}, f"dummy_log.log"]
    
    progress((2,16), desc=f"Getting configuration values")
    time.sleep(0.25)
    # Get config values if not provided
    #config_file = find_file("config.ini")  ##from file_handler.file_utils  ##takes a bit of time to process. #NeedOptimise
    
    config_file = Path("utils") / "config.ini"  ##SMY: speed up sacrificing flexibility
    model_id = model_id if model_id else get_config_value(config_file, "MARKER_CAP", "MODEL_ID")
    openai_base_url = openai_base_url if openai_base_url else get_config_value(config_file, "MARKER_CAP", "OPENAI_BASE_URL")
    openai_image_format = openai_image_format if openai_image_format else get_config_value(config_file, "MARKER_CAP", "OPENAI_IMAGE_FORMAT")
    max_workers = max_workers if max_workers else get_config_value(config_file, "MARKER_CAP", "MAX_WORKERS")
    max_retries = max_retries if max_retries else get_config_value(config_file, "MARKER_CAP", "MAX_RETRIES")
    output_format = output_format if output_format else get_config_value(config_file, "MARKER_CAP", "OUTPUT_FORMAT")
    output_dir_string = output_dir_string if output_dir_string else str(get_config_value(config_file, "MARKER_CAP", "OUTPUT_DIR"))
    use_llm = use_llm if use_llm else get_config_value(config_file, "MARKER_CAP", "USE_LLM")
    page_range = page_range if page_range else get_config_value(config_file,"MARKER_CAP", "PAGE_RANGE")
    weasyprint_dll_directories= weasyprint_dll_directories if weasyprint_dll_directories else None
    config_load_models.weasyprint_libpath = weasyprint_dll_directories  ## Assign user's weasyprint path to Global var
    config_load_models.pdf_files_count = pdf_files_count
    #pooling = True   ##SMY: placeholder
    
    progress((3,16), desc=f"Retrieved configuration values")
    time.sleep(0.25)

    # Create the initargs tuple from the Gradio inputs: # 'files' is an iterable, and handled separately.
    yield gr.update(interactive=False), f"Setting global variables : Initialising init_args", {"process": "Processing files ..."}, f"dummy_log.log"
    progress((4,16), desc=f"Setting global variables : Initialiasing init_args")
    time.sleep(0.25)
    #init_args = ( ...
    
    # set global variables
    from globals import config_load
    #self.pdf_files_count: int = 0
    config_load.provider = provider
    config_load.model_id = model_id
    config_load.hf_provider = hf_provider
    config_load.endpoint = endpoint
    config_load.backend_choice = backend_choice
    config_load.system_message = system_message
    config_load.max_tokens = max_tokens
    config_load.temperature = temperature
    config_load.top_p = top_p
    config_load.stream = stream
    config_load.api_token = api_token
    config_load.openai_base_url = openai_base_url
    config_load.openai_image_format = openai_image_format
    config_load.max_workers = max_workers
    config_load.max_retries = max_retries
    config_load.debug = debug
    #output_format: str = "markdown",
    config_load.output_format = output_format
    config_load.output_dir_string = output_dir_string
    config_load.use_llm = use_llm
    config_load.force_ocr = force_ocr
    config_load.strip_existing_ocr = strip_existing_ocr
    config_load.disable_ocr_math = disable_ocr_math
    config_load.page_range = page_range
    #config_load.weasyprint_dll_directories: str = None,
    config_load.tz_hours = tz_hours
    config_load.pooling = pooling   ## placeholder for ProcessPoolExecutor flag
   
    # 1. create output_dir
    try:
        yield gr.update(interactive=False), f"Creating output_dir ...", {"process": "Processing files ..."}, f"dummy_log.log"
        progress((5,16), desc=f"ProcessPoolExecutor: Creating output_dir")
        time.sleep(0.25)

        #pdf2md_converter.output_dir_string = output_dir_string   ##SMY: attempt setting directly to resolve pool.map iterable

        # Create Marker output_dir in temporary directory where Gradio can access it.  #file_utils.
        output_dir = create_temp_folder(output_dir_string)
        #pdf2md_converter.output_dir = output_dir  ##SMY should now redirect to globals
        config_load.output_dir = output_dir
        
        logger.info(f"✓ output_dir created: ", extra={"output_dir": config_load.output_dir.name, "in": str(config_load.output_dir.parent)})
        yield gr.update(interactive=False), f"Created output_dir ...", {"process": "Processing files ..."}, f"dummy_log.log"
        progress((6,16), desc=f"✓ Created output_dir.")
        time.sleep(0.25)
    except Exception as exc:
            tb = traceback.format_exc()
            tbp = traceback.print_exc()  # Print the exception traceback
            logger.exception("✗ error creating output_dir → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            
            # Update the Gradio UI to improve user-friendly eXperience
            yield gr.update(interactive=True), f"✗ An error occurred creating output_dir: {str(exc)}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  ## return the exception message
            return f"An error occurred creating output_dir: {str(exc)}", f"Error: {exc}", f"Error: {exc}"  ## return the exception message

    # 2. Process file conversion leveraging ProcessPoolExecutor for efficiency 
    results = []  ## Processed files result holder
    logger.log(level=30, msg="Initialising Processing Files ...", extra={"pdf_files": pdf_files, "files_len": len(pdf_files), "model_id": model_id, "output_dir": output_dir_string})  #pdf_files_count
    yield gr.update(interactive=False), f"Initialising Processing Files ...", {"process": "Processing files ..."}, f"dummy_log.log"
    progress((7,16), desc=f"Initialising Processing Files ...")
    time.sleep(0.25)

    try:
        #yield gr.update(interactive=True), f"Pooling file conversion ...", {"process": "Processing files ..."}, f"dummy_log.log"
        progress((8,16), desc=f"Pooling file conversion ...")
        time.sleep(0.25)
        yield gr.update(interactive=False), f"Pooling file conversion ...", {"process": "Processing files ..."}, f"dummy_log.log"
        
        ##SMY: Future: users choose sequential or pooling from Gradio ui
        match pooling:
            case "no_pooling":
                results = get_results_files_conversion(pdf_files, pdf_files_count,progress)
            case "pooling":
                results = get_results_files_conversion_with_pool(pdf_files, pdf_files_count, max_workers, progress)
            case "as_completed":
                results = get_results_files_conversion_with_pool_ascomplete(pdf_files, pdf_files_count, max_workers, progress)
            
        logger.log(level=30, msg="Got Results from files conversion: ", extra={"results": str(results)[:20]}) 
        yield gr.update(interactive=True), f"Got Results from files conversion: [{str(results)[:20]}]", {"process": "Processing files ..."}, f"dummy_log.log"
        progress((9,16), desc=f"Got Results from files conversion")
        time.sleep(0.25)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during Files processing → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        yield gr.update(interactive=True), f"✗ An error occurred during Files Processing → {exc}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  # return the exception message
        return [gr.update(interactive=True), f"✗ An error occurred during files processing → {exc}", {"Error":f"Error: {exc}"}, f"dummy_log.log"]
    
    # 3. Process file conversion results
    try:
        logger.log(level=20, msg="ProcessPoolExecutor pool result:", extra={"results": str(results)})
        progress((12,16), desc="Processing results from files conversion")  ##rekickin
        time.sleep(0.25)
        
        logs = []
        logs_files_images = []

        #logs.extend(results)   ## performant pythonic
        #logs = list[results]  ## 
        logs = [result for result in results]  ## pythonic list comprehension
        # [template]  ## logs : [file , images , filepath, image_path]
        
        #logs_files_images = logs_files.extend(logs_images)  #zip(logs_files, logs_images)   ##SMY: in progress
        logs_count =  0
        #for log in logs:
        for i, log in enumerate(logs):
            logs_files_images.append(log.get("filepath") if is_dict(log) or is_list_of_dicts(logs) else "Error or no file_path")  # isinstance(log, (dict, str))
            logs_files_images.extend(list(image for image in log.get("image_path", "Error or no image_path")))
            i_image_count = log.get("images", 0)
            # Update the Gradio UI to improve user-friendly eXperience
            #yield gr.update(interactive=False), f"Processing files: {logs_files_images[logs_count]}", {"process": "Processing files"}, f"dummy_log.log"
            progress1(0.7, desc=f"Processing result log {i}: {str(log)}")
            logs_count = i+i_image_count
    except Exception as exc:
        tbp = traceback.print_exc()  # Print the exception traceback
        logger.exception("Error during processing results logs → {exc}\n{tbp}", exc_info=True)  # Log the full traceback
        return [gr.update(interactive=True), f"An error occurred during processing results logs: {str(exc)}\n{tbp}", {"Error":f"Error: {exc}"}, f"dummy_log.log"]  ## return the exception message
        #yield gr.update(interactive=True), f"An error occurred during processing results logs: {str(exc)}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  ## return the exception message
    
    
    # 4. Zip Processed Files and images. Insert to first index
    try:  ##from file_handler.file_utils
        progress((13,16), desc="Zipping processed files and images")
        time.sleep(0.25)
        zipped_processed_files = zip_processed_files(root_dir=f"{output_dir}", file_paths=logs_files_images, tz_hours=tz_hours, date_format='%d%b%Y_%H-%M-%S')  #date_format='%d%b%Y'
        logs_files_images.insert(0, zipped_processed_files)

        
        #yield gr.update(interactive=False), f"Processing zip and files: {logs_files_images}", {"process": "Processing files"}, f"dummy_log.log"
        progress((14,16), desc="Zipped processed files and images")
        time.sleep(0.25)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during zipping processed files → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        yield gr.update(interactive=True), f"✗ An error occurred during zipping files → {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  # return the exception message
        return gr.update(interactive=True), f"✗ An error occurred during zipping files → {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  # return the exception message

    
    # 5. Return processed files log
    try:
        progress((15,16), desc="Formatting processed log results")
        time.sleep(0.25)
        
        ## # Convert logs list of dicts to formatted json stringutils.file_utils.
        logs_return_formatted_json_string = process_dicts_data(logs)   #"\n".join(log for log in logs)  ##SMY outputs to gr.JSON component with no need for json.dumps(data, indent=)
        #logs_files_images_return = "\n".join(path for path in logs_files_images)  ##TypeError: sequence item 0: expected str instance, WindowsPath found  
        
        ## # Convert any Path objects to strings, but leave strings as-is
        logs_files_images_return = list(str(path) if isinstance(path, Path) else path for path in logs_files_images)
        logger.log(level=20, msg="File conversion complete. Sending outcome to Gradio:", extra={"logs_files_image_return": str(logs_files_images_return)})  ## debug: FileNotFoundError: [WinError 2] The system cannot find the file specified: 'Error or no image_path'
        
        progress((16,16), desc="Complete processing and formatting file processing results")
        time.sleep(0.25)
        # [templates]
        #outputs=[process_button, log_output, files_individual_JSON, files_individual_downloads],
        #return "\n".join(logs), "\n".join(logs_files_images)    #"\n".join(logs_files)
        
        yield  gr.update(interactive=True), gr.update(value=logs_return_formatted_json_string), gr.update(value=logs_return_formatted_json_string, visible=True), gr.update(value=logs_files_images_return, visible=True)    ##SMY: redundant
        return [gr.update(interactive=True), gr.update(value=logs_return_formatted_json_string), gr.update(value=logs_return_formatted_json_string, visible=True), gr.update(value=logs_files_images_return, visible=True)]
        #yield gr.update(interactive=True), logs_return_formatted_json_string, logs_return_formatted_json_string, logs_files_images_return
        #return [gr.update(interactive=True), logs_return_formatted_json_string, logs_return_formatted_json_string, logs_files_images_return]
        
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during returning result logs → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        yield   gr.update(interactive=True), f"✗ An error occurred during returning result logs→ {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"  # return the exception message
        return [gr.update(interactive=True), f"✗ An error occurred during returning result logs→ {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"dummy_log.log"]  # return the exception message

    #return "\n".join(log for log in logs), "\n".join(str(path) for path in logs_files_images)
    #print(f'logs_files_images: {"\n".join(str(path) for path in logs_files_images)}')

## SMY: to be implemented/refactored AND moved to logic file
'''
def convert_md_to_pdf(file: gr.File | None, folder: str | None) -> list[gr.File]:
    """
    Gradio callback for Markdown → PDF.
    Returns a list of generated PDF files (as Gradio File objects).
    """
    if not file and not folder:
        return []

    md_paths = []

    # Single file
    if file:
        md_path = Path(file.name)
        md_paths.append(md_path)

    # Folder
    if folder:
        try:
            md_paths.extend(collect_markdown_paths(folder))
        except Exception as exc:
            logger.exception("Folder traversal failed.")
            return []

    if not md_paths:
        return []

    output_dir = Path("./generated_pdfs")
    output_dir.mkdir(exist_ok=True)

    pdf_files = md2pdf_converter.batch_convert(md_paths, output_dir)
    # Convert to Gradio File objects
    gr_files = [gr.File(path=str(p)) for p in pdf_files]
    return gr_files
'''


##====================
#Gradio interface moved to gradio_ui.py
#def build_interface() -> gr.Blocks:
#    """
#    Assemble the Gradio Blocks UI.
#    """

if __name__ == '__name__':
    convert_batch()