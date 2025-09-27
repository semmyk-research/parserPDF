# ui/gradio_ui.py
import gradio as gr

from pathlib import Path, WindowsPath
import traceback  ## Extract, format and print information about Python stack traces.
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Union #, Dict, List, Any, Tuple

from huggingface_hub import get_token
import file_handler
import file_handler.file_utils
from utils.config import TITLE, DESCRIPTION, DESCRIPTION_PDF_HTML, DESCRIPTION_PDF, DESCRIPTION_HTML, DESCRIPTION_MD
from utils.utils import is_dict, is_list_of_dicts
from file_handler.file_utils import zip_processed_files, process_dicts_data, collect_pdf_paths, collect_html_paths, collect_markdown_paths, create_outputdir  ## should move to handling file
#from llm.hf_client import HFChatClient  ## SMY: unused. See converters.extraction_converter
from llm.provider_validator import is_valid_provider, suggest_providers
from llm.llm_login import is_loggedin_huggingface, login_huggingface

from converters.extraction_converter import DocumentConverter as docconverter  #DocumentExtractor #as docextractor
from converters.pdf_to_md import PdfToMarkdownConverter, init_worker
#from converters.md_to_pdf import MarkdownToPdfConverter
#from converters.html_to_md import HtmlToMarkdownConverter  ##SMY: PENDING: implementation

from file_handler.file_utils import find_file

from utils.get_config import get_config_value
from utils.logger import get_logger

logger = get_logger(__name__)   ##NB: setup_logging()  ## set logging

# Instantiate converters class once – they are stateless
pdf2md_converter = PdfToMarkdownConverter()
#html2md_converter = HtmlToMarkdownConverter()
#md2pdf_converter = MarkdownToPdfConverter()

# pool executor to convert files called by Gradio
##SMY: TODO: future: refactor to gradio_process.py
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
    api_token: str,
    #max_workers: int,
    #max_retries: int,
    openai_base_url: str = "https://router.huggingface.co/v1",
    openai_image_format: Optional[str] = "webp",
    max_workers: Optional[int] = 4,
    max_retries: Optional[int] = 2,
    output_format: str = "markdown",
    #output_dir: Optional[Union[str, Path]] = "output_dir",
    output_dir_string: str = "output_dir_default",
    use_llm: bool = False,   #Optional[bool] = False,  #True,
    page_range: str = None,  #Optional[str] = None,
    tz_hours: str = None,
    ): #-> str:
    """
    Handles the conversion process using multiprocessing.
    Spins up a pool and converts all uploaded files in parallel.
    Aggregates per-file logs into one string.
    Receives Gradio component values, starting with the list of uploaded file paths
    """

    # explicitly wrap file object in a list
    #pdf_files = pdf_files_wrap(pdf_files)   ##Flag:  deprecation

    # Update the Gradio UI to improve user-friendly eXperience
    #outputs=[process_button, log_output, files_individual_JSON, files_individual_downloads],
    yield gr.update(interactive=False), f"Processing files...", {"process": "Processing files"}, f"__init__.py"
    
    ## debug
    #logger.log(level=30, msg="pdf_files_inputs", extra={"input_arg[0]:": pdf_files[0]})

    #if not files:
    if not pdf_files or pdf_files is None:  ## Check if files is None. This handles the case where no files are uploaded.
        logger.log(level=30, msg="Initialising ProcessPool: No files uploaded.", extra={"pdf_files": pdf_files, "files_len": pdf_files_count})
        #outputs=[log_output, files_individual_JSON, files_individual_downloads],
        return gr.update(interactive=True), "Initialising ProcessPool: No files uploaded.", {"Upload":"No files uploaded"}, f"__init__.py"
    
    # Get config values if not provided
    config_file = find_file("config.ini")  ##from file_handler.file_utils
    model_id = get_config_value(config_file, "MARKER_CAP", "MODEL_ID") if not model_id else model_id
    openai_base_url = get_config_value(config_file, "MARKER_CAP", "OPENAI_BASE_URL") if not openai_base_url else openai_base_url
    openai_image_format = get_config_value(config_file, "MARKER_CAP", "OPENAI_IMAGE_FORMAT") if not openai_image_format else openai_image_format
    max_workers = get_config_value(config_file, "MARKER_CAP", "MAX_WORKERS") if not max_workers else max_workers
    max_retries = get_config_value(config_file, "MARKER_CAP", "MAX_RETRIES") if not max_retries else max_retries
    output_format = get_config_value(config_file, "MARKER_CAP", "OUTPUT_FORMAT") if not output_format else output_format
    output_dir_string = str(get_config_value(config_file, "MARKER_CAP", "OUTPUT_DIR") if not output_dir_string else output_dir_string)
    use_llm = get_config_value(config_file, "MARKER_CAP", "USE_LLM") if not use_llm else use_llm
    page_range = get_config_value(config_file,"MARKER_CAP", "PAGE_RANGE") if not page_range else page_range
         
    # Create the initargs tuple from the Gradio inputs: # 'files' is an iterable, and handled separately.
    init_args = (
            provider,            
            model_id,
            #base_url,
            hf_provider,
            endpoint,
            backend_choice,
            system_message,
            max_tokens,
            temperature,
            top_p,
            stream,
            api_token,
            openai_base_url,
            openai_image_format,
            max_workers,
            max_retries,
            output_format,
            output_dir_string,
            use_llm,
            page_range,
        )
    
    #global docextractor   ##SMY: deprecated.
    try:
        ##SMY: might deprecate. To replace with oauth login from Gradio ui or integrate cleanly.
        login_huggingface(api_token)  ## attempt login if not already logged in. NB: HF CLI login prompt would not display in Process Worker.
        
        if is_loggedin_huggingface() and (api_token is None or api_token == ""):
            api_token = get_token()
        else:
            login_huggingface()
        # login: Update the Gradio UI to improve user-friendly eXperience
        yield gr.update(interactive=False), f"login to HF: Processing files...", {"process": "Processing files"}, f"__init__.py"
        
    except Exception as exc:  # Catch all exceptions
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during login_huggingface → {exc}\n{tb}", exc_info=True) # Log the full traceback
        return gr.update(interactive=True), f"✗ An error occurred during login_huggingface → {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"__init__.py"  # return the exception message

    try:
        # Create a pool with init_worker initialiser
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker,
            initargs=init_args
        ) as pool:
            #global docextractor
            logger.log(level=30, msg="Initialising ProcessPool: pool:", extra={"pdf_files": pdf_files, "files_len": len(pdf_files), "model_id": model_id, "output_dir": output_dir_string})  #pdf_files_count
            
            # Map the files (pdf_files) to the conversion function (pdf2md_converter.convert_file)
            # The 'docconverter' argument is implicitly handled by the initialiser
            
            #futures = [pool.map(pdf2md_converter.convert_files, f) for f in pdf_files]
            #logs = [f.result() for f in as_completed(futures)]
            #futures = [pool.submit(pdf2md_converter.convert_files, file) for file in pdf_files]
            #logs = [f.result() for f in futures]
            
            try:
                pdf2md_converter.output_dir_string = output_dir_string   ##SMY: attempt setting directly to resolve pool.map iterable
                #result_convert = pool.map(pdf2md_converter.convert_files, pdf_files, max_retries)
                results = pool.map(pdf2md_converter.convert_files, pdf_files)  ##SMY iterables  #output_dir_string)
            except Exception as exc:
                # Raise the exception to stop the Gradio app: exception to halt execution
                logger.exception("Error during pooling file conversion", exc_info=True)  # Log the full traceback
                traceback.print_exc()  # Print the exception traceback
                #return f"An error occurred during pool.map: {str(exc)}", f"Error: {exc}", f"Error: {exc}"  ## return the exception message
                yield gr.update(interactive=True), f"An error occurred during pool.map: {str(exc)}", {"Error":f"Error: {exc}"}, f"__init__.py"  ## return the exception message
    
            #'''
            try:
                logger.log(level=20, msg="ProcessPoolExecutor pool result:", extra={"results": str(results)})
                logs = []
                logs_files_images = []
                #logs.extend(results)   ## performant pythonic
                #logs = list[results]  ## 
                logs = [result for result in results]  ## pythonic list comprehension
                ## logs : [file , images , filepath, image_path]
                
                #logs_files_images = logs_files.extend(logs_images)  #zip(logs_files, logs_images)   ##SMY: in progress
                logs_count =  0
                #for log in logs:
                for i, log in enumerate(logs):
                    logs_files_images.append(log.get("filepath") if is_dict(log) or is_list_of_dicts(logs) else "Error or no file_path")  # isinstance(log, (dict, str))
                    logs_files_images.extend(list(image for image in log.get("image_path", "Error or no image_path")))
                    i_image = log.get("images", 0)
                    # Update the Gradio UI to improve user-friendly eXperience
                    yield gr.update(interactive=False), f"Processing files: {logs_files_images[logs_count]}", {"process": "Processing files"}, f"__init__.py"
                    logs_count = i+i_image
                
                #logs_files_images.append(logs_filepath) ## to del
                #logs_files_images.extend(logs_images)   ## to del
            except Exception as exc:
                logger.exception("Error during processing results logs → {exc}\n{tb}", exc_info=True)  # Log the full traceback
                traceback.print_exc()  # Print the exception traceback
                #return f"An error occurred during processing results logs: {str(exc)}\n{tb}", f"Error: {exc}", f"Error: {exc}"  ## return the exception message
                yield gr.update(interactive=True), f"An error occurred during processing results logs: {str(exc)}\n{tb}", {"Error":f"Error: {exc}"}, f"__init__.py"  ## return the exception message
            
            #'''
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during ProcessPoolExecutor → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        #return gr.update(interactive=True), f"✗ An error occurred during ProcessPoolExecutor→ {exc}\n{tb}", f"Error: {exc}", f"Error: {exc}"  # return the exception message
        yield gr.update(interactive=True), f"✗ An error occurred during ProcessPoolExecutor→ {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"__init__.py"  # return the exception message

    '''
    logger.log(level=20, msg="ProcessPoolExecutor pool result:", extra={"results": str(results)})
    logs = []
    #logs.extend(results)   ## performant pythonic
    #logs = list[results]  ## 
    logs = [result for result in results]  ## pythonic list comprehension
    '''

    # Zip Processed md Files and images. Insert to first index
    try:  ##from file_handler.file_utils
        zipped_processed_files = zip_processed_files(root_dir=f"data/{output_dir_string}", file_paths=logs_files_images, tz_hours=tz_hours, date_format='%d%b%Y')
        logs_files_images.insert(0, zipped_processed_files)
        #logs_files_images.insert(1, "====================")
        yield gr.update(interactive=False), f"Processing zip and files: {logs_files_images}", {"process": "Processing files"}, f"__init__.py"
    
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during zipping processed files → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        #return gr.update(interactive=True), f"✗ An error occurred during zipping files → {exc}\n{tb}", f"Error: {exc}", f"Error: {exc}"  # return the exception message
        yield gr.update(interactive=True), f"✗ An error occurred during zipping files → {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"__init__.py"  # return the exception message

    
    # Return processed files log
    try:
        ## # Convert logs list of dicts to formatted json string
        logs_return_formatted_json_string = file_handler.file_utils.process_dicts_data(logs)   #"\n".join(log for log in logs)  ##SMY outputs to gr.JSON component with no need for json.dumps(data, indent=)
        #logs_files_images_return = "\n".join(path for path in logs_files_images)  ##TypeError: sequence item 0: expected str instance, WindowsPath found
        
        ##convert the List of Path objects to List of string for gr.Files output
        #logs_files_images_return = list(str(path) for path in logs_files_images)  
        
        ## # Convert any Path objects to strings, but leave strings as-is
        logs_files_images_return = list(str(path) if isinstance(path, Path) else path for path in logs_files_images)
        logger.log(level=20, msg="File conversion complete. Sending outcome to Gradio:", extra={"logs_files_image_return": str(logs_files_images_return)})  ## debug: FileNotFoundError: [WinError 2] The system cannot find the file specified: 'Error or no image_path'
        
        #outputs=[process_button, log_output, files_individual_JSON, files_individual_downloads],
        #return "\n".join(logs), "\n".join(logs_files_images)    #"\n".join(logs_files)
        #return logs_return_formatted_json_string, logs_return_formatted_json_string, logs_files_images_return
        #return gr.update(interactive=True), gr.update(value=logs_return_formatted_json_string), gr.update(value=logs_return_formatted_json_string, visible=True), gr.update(value=logs_files_images_return, visible=True)
        yield  gr.update(interactive=True), gr.update(), gr.update(visible=True), gr.update(visible=True)
        yield gr.update(interactive=True), logs_return_formatted_json_string, logs_return_formatted_json_string, logs_files_images_return
        
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception(f"✗ Error during returning result logs → {exc}\n{tb}" , exc_info=True)  # Log the full traceback
        #traceback.print_exc()  # Print the exception traceback
        return gr.update(interactive=True), f"✗ An error occurred during returning result logs→ {exc}\n{tb}", {"Error":f"Error: {exc}"}, f"__init__.py"  # return the exception message


    #return "\n".join(log for log in logs), "\n".join(str(path) for path in logs_files_images)
    #print(f'logs_files_images: {"\n".join(str(path) for path in logs_files_images)}')
        
# files wrapping into list  ##SMY: Flagged for deprecation
def pdf_files_wrap(files: list[str]):
    # explicitly wrap file object in a list
    return [files] if not isinstance(files, list) else files
    #return [files]

##====================
## SMY: moved to logic file: See pdf_to_md.py. Currently unused
def convert_pdfs_to_md(file: gr.File | None, folder: str | None) -> dict:
    """
    Gradio callback for PDF → Markdown.
    Accepts either a single file or a folder path (recursively).
    Leverages Marker, a pipeline of deep learning models, for conversion
    Returns a dictionary of filename → Markdown string.
    """
    if not file and not folder:
        return {"error": "Please provide a PDF file or a folder."}

    pdf_paths = []

    # Single file
    if file:
        pdf_path = Path(file.name)
        pdf_paths.append(pdf_path)

    # Folder (recursively)
    if folder:
        try:
            pdf_paths.extend(collect_pdf_paths(folder))
        except Exception as exc:
            logger.exception("Folder traversal failed.")
            return {"error": str(exc)}

    if not pdf_paths:
        return {"error": "No PDF files found."}

    results = pdf2md_converter.batch_convert(pdf_paths)
    # Gradio expects a dict of {filename: content}
    return results

## SMY: to be implemented AND to refactor and moved to logic file
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


## SMY: to refactor and moved to logic file. Currently unused
'''
def convert_htmls_to_md(file: gr.File | None, folder: str | None) -> dict:
    """
    Gradio callback for HTML → Markdown.
    Accepts either a single file or a folder path (recursively).
    Returns a dictionary of filename → Markdown string.
    """
    if not file and not folder:
        return {"error": "Please provide a HTML file or a folder."}

    html_paths = []

    # Single file
    if file:
        html_path = Path(file.name)
        html_paths.append(html_path)

    # Folder (recursively)
    if folder:
        try:
            html_paths.extend(collect_html_paths(folder))
        except Exception as exc:
            logger.exception("Folder traversal failed.")
            return {"error": str(exc)}

    if not html_paths:
        return {"error": "No HTML files found."}

    results = html2md_converter.batch_convert(html_paths)
    # Gradio expects a dict of {filename: content}
    return results
'''

##====================

def build_interface() -> gr.Blocks:
    """
    Assemble the Gradio Blocks UI.
    """
        
    # Use custom CSS to style the file component
    custom_css = """
    .file-or-directory-area {
        border: 2px dashed #ccc;
        padding: 20px;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .file-or-directory-area:hover {
        border-color: #007bff;
        background-color: #f8f9fa;
    }
    .gradio-upload-btn {
        margin-top: 10px;
    }
    """

    ##SMY: flagged; to move to file_handler.file_utils
    def is_file_with_extension(path_obj: Path) -> bool:
        """
        Checks if a pathlib.Path object is a file and has a non-empty extension.
        """
        path_obj = path_obj if isinstance(path_obj, Path) else Path(path_obj) if isinstance(path_obj, str) else None
        return path_obj.is_file() and bool(path_obj.suffix)

    ##SMY: flagged; to move to file_handler.file_utils
    def accumulate_files(uploaded_files, current_state):
        """
        Accumulates newly uploaded files with the existing state.
        """
        # Initialize state if it's the first run
        if current_state is None:
            current_state = []
        
        # If no files were uploaded in this interaction, return the current state unchanged
        if not uploaded_files:
            return current_state, f"No new files uploaded. Still tracking {len(current_state)} file(s)."
        
        # Get the temporary paths of the newly uploaded files
        # call is_file_with_extension to check if pathlib.Path object is a file and has a non-empty extension
        new_file_paths = [f.name for f in uploaded_files if is_file_with_extension(Path(f.name))]  #Path(f.name) and Path(f.name).is_file() and bool(Path(f.name).suffix)]  #Path(f.name).suffix.lower() !=""]

        # Concatenate the new files with the existing ones in the state
        updated_files = current_state + new_file_paths
        updated_filenames = [Path(f).name for f in updated_files]
        
        # Return the updated state and a message to the user
        #file_info = "\n".join(updated_files)
        filename_info = "\n".join(updated_filenames)
        #message = f"Accumulated {len(updated_files)} file(s) total.\n\nAll file paths:\n{file_info}"
        message = f"Accumulated {len(updated_files)} file(s) total: \n{filename_info}"
        
        return updated_files, message

    def clear_state():
        """
        Clears the accumulated state of uloaded file list, output textbox, files and directory upload.
        """
        return [], "Files list cleared.", [], []

    # with gr.Blocks(title=TITLE) as demo
    with gr.Blocks(title=TITLE, css=custom_css) as demo:
        gr.Markdown(f"## {DESCRIPTION}")

        # Clean UI: Model parameters hidden in collapsible accordion
        with gr.Accordion("⚙️ LLM Model Settings", open=False):
            gr.Markdown(f"#### **Backend Configuration**")
            system_message = gr.Textbox(
                label="System Message",
                lines=2,
            )
            with gr.Row():
                provider_dd = gr.Dropdown(
                    choices=["huggingface", "openai"],
                    label="Provider",
                    value="huggingface",
                    #allow_custom_value=True,
                )
                backend_choice = gr.Dropdown(
                    choices=["model-id", "provider", "endpoint"],
                    label="HF Backend Choice",
                )  ## SMY: ensure HFClient maps correctly 
                model_tb = gr.Textbox(
                    label="Model ID",
                    value="meta-llama/Llama-4-Maverick-17B-128E-Instruct",  #image-Text-to-Text  #"openai/gpt-oss-120b",  ##Text-to-Text
                )
                endpoint_tb = gr.Textbox(
                    label="Endpoint",
                    placeholder="Optional custom endpoint",
                )
            with gr.Row():
                max_token_sl = gr.Slider(
                    label="Max Tokens",
                    minimum=1,
                    maximum=131172,  #65536,  #32768,  #16384,  #8192,
                    value=1024,  #512,
                    step=1,
                )
                temperature_sl = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,  #0.01
                )
                top_p_sl = gr.Slider(
                    label="Top-p",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,  #0.01
                )
                with gr.Column():
                    stream_cb = gr.Checkbox(
                        label="LLM Streaming",
                        value=False,
                    )
                    #tz_hours_tb = gr.Textbox(value=None, label="TZ Hours", placeholder="Timezone in numbers", max_lines=1,)
                    tz_hours_num = gr.Number(label="TZ Hours", placeholder="Timezone in numbers", min_width=5,)
            with gr.Row():
                api_token_tb = gr.Textbox(
                    label="API Token [OPTIONAL]",
                    type="password",
                    placeholder="hf_xxx or openai key"
                )
                hf_provider_dd = gr.Dropdown(
                    choices=["fireworks-ai", "together-ai", "openrouter-ai", "hf-inference"],
                    value="fireworks-ai",
                    label="Provider",
                    allow_custom_value=True,  # let users type new providers as they appear
                )

        # Validate provider on change; warn but allow continue
        def on_provider_change(provider_value: str):
            if not provider_value:
                return
            if not is_valid_provider(provider_value):
                sug = suggest_providers(provider_value)
                extra = f" Suggestions: {', '.join(sug)}." if sug else ""
                gr.Warning(
                    f"Provider not on HF provider list. See https://huggingface.co/docs/inference-providers/index.{extra}"
                )
        hf_provider_dd.change(on_provider_change, inputs=hf_provider_dd, outputs=None)

        # Clean UI: Model parameters hidden in collapsible accordion
        with gr.Accordion("⚙️ Marker Settings", open=False):
            gr.Markdown(f"#### **Marker Configuration**")
            with gr.Row():
                openai_base_url_tb = gr.Textbox(
                    label="OpenAI Base URL: Default HuggingFace",
                    value="https://router.huggingface.co/v1",
                    lines=1,
                    max_lines=1,
                )
                openai_image_format_dd = gr.Dropdown(
                    choices=["webp", "png", "jpeg"],
                    label="OpenAI Image Format",
                    value="webp",
                )
                output_format_dd = gr.Dropdown(
                    choices=["markdown", "html"],  #, "json", "chunks"],  ##SMY: To be enabled later
                    #choices=["markdown", "html", "json", "chunks"],
                    label="Output Format",
                    value="markdown",
                )
                output_dir_tb = gr.Textbox(
                    label="Output Directory",
                    value="output_dir",  #"output_md",
                    lines=1,
                    max_lines=1,
                )
            with gr.Row():
                max_workers_sl = gr.Slider(
                    label="Max Worker",
                    minimum=1,
                    maximum=7,
                    value=4,
                    step=1  
                )
                max_retries_sl = gr.Slider(
                    label="Max Retry",
                    minimum=1,
                    maximum=3,
                    value=2,
                    step=1  #0.01
                )
                use_llm_cb = gr.Checkbox(
                    label="Use LLM for Marker conversion",
                    value=False
                )
                page_range_tb = gr.Textbox(
                    label="Page Range (Optional)",
                    placeholder="Example: 0,1-5,8,12-15",
                    lines=1,
                    max_lines=1,
                )

        # Initialise gr.State
        state_max_workers = gr.State(4)  #max_workers_sl,
        state_max_retries = gr.State(2) #max_retries_sl,
        state_tz_hours    = gr.State(value=None)

        def update_state_stored_value(new_component_input):
            """ Updates stored state: use for max_workers and max_retries """
            return new_component_input
        
        # Update gr.State values on slider components change. NB: initial value of `gr.State` must be able to be deepcopied
        max_workers_sl.change(update_state_stored_value, inputs=max_workers_sl, outputs=state_max_workers)
        max_retries_sl.change(update_state_stored_value, inputs=max_retries_sl, outputs=state_max_retries)
        tz_hours_num.change(update_state_stored_value, inputs=tz_hours_num, outputs=state_tz_hours)


        with gr.Accordion("🤗 HuggingFace Client Logout", open=True):  #, open=False):
            # Logout controls
            '''def do_logout():
                try:
                    #ok = docextractor.client.logout()
                    ok = docconverter.client.logout()
                    # Reset token textbox on successful logout
                    #msg = "✅ Logged out of HuggingFace and cleared tokens. Remember to log out of HuggingFace completely." if ok else "⚠️ Logout failed."
                    msg = "✅ Session Cleared. Remember to browser." if ok else "⚠️ Logout failed."
                    return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session")
                except AttributeError:
                    msg = "⚠️ Logout."
                    return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session", interactive=False)
            '''
            def do_logout_hf():
                try:
                    ok = docconverter.client.logout()
                    # Reset token textbox on successful logout
                    msg = "✅ Session Cleared. Remember to close browser." if ok else "⚠️ Logout & Session Cleared"
                    #return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session", interactive=False)
                    #return msg
                    yield msg
                except AttributeError:
                    msg = "⚠️ Logout. No HF session"
                    #return msg
                    yield msg
                
            def custom_do_logout():
                #do_logout()
                #return gr.update(value="Sign in to HuggingFace 🤗")
                msg = do_logout_hf()
                #return gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value=""), gr.update(visible=True, value=msg)
                yield gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value=""), gr.update(visible=True, value=msg)
            
            logout_status = gr.Markdown(visible=False)
            with gr.Row():
                hf_login_logout_btn = gr.LoginButton(value="Sign in to HuggingFace 🤗", logout_value="Clear Session & Logout of HF: ({})", variant="huggingface")
                #logout_btn = gr.Button("Logout from session and Hugging Face (inference) Client", variant="stop", )

            #hf_login_logout_btn.click(fn=custom_do_logout, inputs=None, outputs=hf_login_logout_btn)
            hf_login_logout_btn.click(fn=custom_do_logout, inputs=None, outputs=[hf_login_logout_btn, api_token_tb, logout_status])
            #logout_btn.click(fn=do_logout, inputs=None, outputs=[api_token_tb, logout_status, hf_login_logout_btn, logout_btn])

        
        # The gr.State component to hold the accumulated list of files
        uploaded_file_list = gr.State([])   ##NB: initial value of `gr.State` must be able to be deepcopied

        # --- PDF & HTML → Markdown tab ---
        with gr.Tab(" 📄 PDF & HTML ➜ Markdown"):
            gr.Markdown(f"#### {DESCRIPTION_PDF_HTML}")

            ### flag4deprecation  #earlier implementation
            '''
            pdf_files = gr.File(
                label="Upload PDF, HTML or PDF and HTMLfiles",
                file_count="directory", ## handle directory and files upload #"multiple",
                type="filepath",
                file_types=["pdf", ".pdf"],
                #size="small",
            )
            pdf_files_count = gr.TextArea(label="Files Count", interactive=False, lines=1)
            with gr.Row():
                btn_pdf_count = gr.Button("Count Files")
                #btn_pdf_upload = gr.UploadButton("Upload files")
                btn_pdf_convert = gr.Button("Convert PDF(s)")
            '''

            with gr.Column(elem_classes=["file-or-directory-area"]):
                with gr.Row():
                    file_btn = gr.UploadButton(
                    #file_btn = gr.File(
                        label="Upload Multiple Files",
                        file_count="multiple",
                        file_types=["file"],
                        #height=25,  #"sm",
                        size="sm",
                        elem_classes=["gradio-upload-btn"]
                    )
                    dir_btn = gr.UploadButton(
                    #dir_btn = gr.File(
                        label="Upload a Directory",
                        file_count="directory",
                        #file_types=["file"],  #Warning: The `file_types` parameter is ignored when `file_count` is 'directory'
                        #height=25,  #"0.5",
                        size="sm",
                        elem_classes=["gradio-upload-btn"]
                    )
            with gr.Accordion("Display uploaded", open=True):
                # Displays the accumulated file paths
                output_textbox = gr.Textbox(label="Accumulated Files", lines=3) #, max_lines=4)  #10
            
            with gr.Row():
                process_button = gr.Button("Process All Uploaded Files", variant="primary")
                clear_button = gr.Button("Clear All Uploads", variant="secondary")

            # Event handler for the multiple file upload button
            file_btn.upload(
                fn=accumulate_files,
                inputs=[file_btn, uploaded_file_list],
                outputs=[uploaded_file_list, output_textbox]
            )

            # Event handler for the directory upload button
            dir_btn.upload(
                fn=accumulate_files,
                inputs=[dir_btn, uploaded_file_list],
                outputs=[uploaded_file_list, output_textbox]
            )

            # Event handler for the "Clear" button
            clear_button.click(
                fn=clear_state,
                inputs=None,
                outputs=[uploaded_file_list, output_textbox, file_btn, dir_btn],
            )

        # --- PDF → Markdown tab ---
        with gr.Tab(" 📄 PDF ➜ Markdown (Flag for DEPRECATION)", interactive=False, visible=True):  #False
            gr.Markdown(f"#### {DESCRIPTION_PDF}")

            files_upload_pdf = gr.File(
                label="Upload PDF files",
                file_count="directory", ## handle directory and files upload #"multiple",
                type="filepath",
                file_types=["pdf", ".pdf"],
                #size="small",
            )
            files_count = gr.TextArea(label="Files Count", interactive=False, lines=1)  #pdf_files_count
            with gr.Row():
                btn_pdf_count = gr.Button("Count Files")
                #btn_pdf_upload = gr.UploadButton("Upload files")
                btn_pdf_convert = gr.Button("Convert PDF(s)")
        
        # --- 📃 HTML → Markdown tab ---
        with gr.Tab("🕸️ HTML ➜ Markdown: (Flag for DEPRECATION)", interactive=False, visible=False):
            gr.Markdown(f"#### {DESCRIPTION_HTML}")

            files_upload_html = gr.File(
                label="Upload HTML files",
                file_count="multiple",
                type="filepath",
                file_types=["html", ".html", "htm", ".htm"]
            )
            #btn_html_convert = gr.Button("Convert HTML(s)")
            html_files_count = gr.TextArea(label="Files Count", interactive=False, lines=1)
            with gr.Row():
                btn_html_count = gr.Button("Count Files")
                #btn_pdf_upload = gr.UploadButton("Upload files")
                btn_html_convert = gr.Button("Convert PDF(s)")


        # --- Markdown → PDF tab ---
        with gr.Tab("PENDING: Markdown ➜ PDF", interactive=False):
            gr.Markdown(f"#### {DESCRIPTION_MD}")

            md_files = gr.File(
                label="Upload Markdown files",
                file_count="multiple",
                type="filepath",
                file_types=["md", ".md"]
            )
            btn_md_convert = gr.Button("Convert Markdown to PDF)")
            output_pdf = gr.Gallery(label="Generated PDFs", elem_id="pdf_gallery")

            '''
            md_input = gr.File(label="Upload a single Markdown file", file_count="single")
            md_folder_input = gr.Textbox(
                label="Or provide a folder path (recursively)",
                placeholder="/path/to/folder",
            )
            convert_md_btn = gr.Button("Convert Markdown to PDF")
            output_pdf = gr.Gallery(label="Generated PDFs", elem_id="pdf_gallery")

            convert_md_btn.click(
                fn=convert_md_to_pdf,
                inputs=[md_input, md_folder_input],
                outputs=output_pdf,
            )
            '''

        # A Files component to display individual processed files as download links
        with gr.Accordion("⏬ View and Download processed files", open=True):  #, open=False
            processed_file_state = gr.State([])

            ##SMY: future
            zip_btn = gr.DownloadButton("Download Zip file of all processed files", visible=False)   #.Button()
            
            # Placeholder to download zip file of processed files
            download_zip_file = gr.File(label="Download processed Files (ZIP)", interactive=False, visible=False)  #, height="1"

            with gr.Row():
                files_individual_JSON = gr.JSON(label="Serialised JSON list", max_height=250, visible=False)
                files_individual_downloads = gr.Files(label="Individual Processed Files", visible=False)

        ## Displays processed file paths
        with gr.Accordion("View processing log", open=True): #open=False):
            log_output = gr.Textbox(
                label="Conversion Logs",
                lines=5,
                #max_lines=25,
                #interactive=False
            )
        
        # file inputs
        ## [wierd] NB: inputs_arg is a list of Gradio component objects, not the values of those components.
        ## inputs_arg variable captures the state of these components at the time the list is created. 
        ## When btn_convert.click() is called later, it uses the list as it was initially defined
        ##
        ## SMY: Gradio component values are not directly mutable.
        ## Instead, you should pass the component values to a function,
        ## and then use the return value of the function to update the component.
        ## Discarding for now. #//TODO: investigate further.
        ## SMY: Solved: using gr.State 
        inputs_arg = [
            #pdf_files,
            ##pdf_files_wrap(pdf_files),  # wrap pdf_files in a list (if not already)
            uploaded_file_list,
            files_count,  #pdf_files_count,
            provider_dd,
            model_tb,
            hf_provider_dd,
            endpoint_tb,
            backend_choice,
            system_message,
            max_token_sl,
            temperature_sl,
            top_p_sl,
            stream_cb,
            api_token_tb,
            #gr.State(4),   # max_workers
            #gr.State(3),    # max_retries
            openai_base_url_tb,
            openai_image_format_dd,
            state_max_workers, #gr.State(4),  #max_workers_sl,
            state_max_retries, #gr.State(2), #max_retries_sl,
            output_format_dd,
            output_dir_tb,
            use_llm_cb,
            page_range_tb,
            tz_hours_num,
        ]

        ## debug
        #logger.log(level=30, msg="About to execute btn_pdf_convert.click", extra={"files_len": pdf_files_count, "pdf_files": pdf_files})
        
        try:
            #logger.log(level=30, msg="input_arg[0]: {input_arg[0]}")
            process_button.click(
            #pdf_files.upload( 
                fn=convert_batch,
                inputs=inputs_arg,
                outputs=[process_button, log_output, files_individual_JSON, files_individual_downloads],
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"✗ Error during process_button.click → {exc}\n{tb}", exc_info=True)
            msg = "✗ An error occurred during process_button.click"  # →
            #return f"✗ An error occurred during process_button.click → {exc}\n{tb}"
            return gr.update(interactive=True), f"{msg} → {exc}\n{tb}", f"{msg} → {exc}", f"{msg} → {exc}"

        ##gr.File .upload() event, fire only after a file has been uploaded
        # Event handler for the pdf file upload button
        files_upload_pdf.upload(
            fn=accumulate_files,
            inputs=[files_upload_pdf, uploaded_file_list],
            outputs=[uploaded_file_list, log_output]
        )
        #inputs_arg[0] = files_upload
        btn_pdf_convert.click(
        #pdf_files.upload( 
            fn=convert_batch,
            outputs=[btn_pdf_convert, log_output, files_individual_JSON, files_individual_downloads],
            inputs=inputs_arg, 
        ) 
        #    )

        # reuse the same business logic for HTML tab
        # Event handler for the pdf file upload button
        files_upload_html.upload(
            fn=accumulate_files,
            inputs=[files_upload_html, uploaded_file_list],
            outputs=[uploaded_file_list, log_output]
        )
        #inputs_arg[0] = html_files
        btn_html_convert.click(
            fn=convert_batch,
            inputs=inputs_arg,
            outputs=[btn_html_convert,log_output, files_individual_JSON, files_individual_downloads]
        )

        def get_file_count(file_list):
            """
            Counts the number of files in the list.

            Args:
                file_list (list): A list of temporary file objects.
            Returns:
                str: A message with the number of uploaded files.
            """
            if file_list:
                return f"{len(file_list)}", f"Upload: {len(file_list)} files: \n {file_list}"  #{[pdf_files.value]}"
            else:
                return "No files uploaded.", "No files uploaded."        # Count files button
        
        btn_pdf_count.click(
            fn=get_file_count,
            inputs=[files_upload_pdf],
            outputs=[files_count, log_output]
        )
        btn_html_count.click(
            fn=get_file_count,
            inputs=[files_upload_html],
            outputs=[html_files_count, log_output]
        )        
        
        # Validate files upload on change; warn but allow continue
        def on_pdf_files_change(pdf_files_value: list[str]):
            # explicitly wrap file object in a list
            pdf_files_value = pdf_files_wrap(pdf_files_value)
            #if not isinstance(pdf_files_value, list):
            #    pdf_files_value = [pdf_files_value]

            pdf_files_path = [file.name for file in pdf_files_value]
            pdf_files_len = len(pdf_files_value)  #len(pdf_files_path)
            if pdf_files_value:
                #return            
                return pdf_files_path, pdf_files_len
        #pdf_files.change(on_pdf_files_change, inputs=pdf_files, outputs=[log_output, pdf_files_count])  #, postprocess=False)  ##debug


    return demo

