# ui/gradio_ui.py

import gradio as gr
from ui.gradio_process import convert_batch
from globals import config_load 

from llm.provider_validator import is_valid_provider, suggest_providers
from converters.extraction_converter import DocumentConverter as docconverter  #DocumentExtractor #as docextractor

from utils.config import TITLE, DESCRIPTION, DESCRIPTION_PDF_HTML, DESCRIPTION_PDF, DESCRIPTION_HTML, DESCRIPTION_MD
from utils.file_utils import accumulate_files, is_file_with_extension

import traceback  ## Extract, format and print information about Python stack traces.
from utils.logger import get_logger

logger = get_logger(__name__)   ##NB: setup_logging()  ## set logging

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

    ##SMY: flagged; to move to file_handler.file_utils #accumulate_files()
    
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
                    value=8192,  #1024,  #512,
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

        # Clean UI: Model parameters hidden in collapsible accordion
        with gr.Accordion("⚙️ Marker Converter Settings", open=False):
            gr.Markdown(f"#### **Marker Configuration**")
            with gr.Row():
                openai_base_url_tb = gr.Textbox(
                    label="OpenAI Base URL",
                    info = "default HuggingFace",
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
                    choices=["markdown", "html", "json"],  #, "json", "chunks"],  ##SMY: To be enabled later
                    #choices=["markdown", "html", "json", "chunks"],
                    label="Output Format",
                    value="markdown",
                )
            with gr.Row():
                #pooling_cb = gr.Checkbox(
                pooling_dd = gr.Dropdown(
                    label="Pool: multiprocessing",
                    info="Enable for high # of files [Beware!]",
                    value="no_pooling",  #True,  #False
                    choices=["no_pooling", "pooling", "as_completed"]
                )
                max_workers_sl = gr.Slider(
                    label="Max Worker",
                    minimum=1,
                    maximum=4,
                    value=3,
                    step=1  
                )
                max_retries_sl = gr.Slider(
                    label="Max Retry",
                    minimum=1,
                    maximum=3,
                    value=2,
                    step=1  #0.01
                )
                output_dir_tb = gr.Textbox(
                    label="Output Directory",
                    value="output_dir",  #"output_md",
                    lines=1,
                    max_lines=1,
                )
            with gr.Row():
                with gr.Column():
                    debug_cb = gr.Checkbox(
                        label="Run in debug mode. Not recommended",
                        value=False,  #True,
                    )
                    use_llm_cb = gr.Checkbox(
                        label="Use LLM for Marker conversion",
                        value=False
                    )
                    force_ocr_cb = gr.Checkbox(
                        label="Force OCR on all pages. (Beware: extended processing time)",
                        value=False,  #True,
                    )
                #with gr.Column():
                    strip_existing_ocr_cb = gr.Checkbox(
                        label="strip embedded OCR text, re-run OCR",
                        value=False
                    )
                    disable_ocr_math_cb = gr.Checkbox(
                        label="OCR: disable math - no inline math",
                        value=False,
                    )
                with gr.Column():
                    page_range_tb = gr.Textbox(
                        label="Page Range (Optional)",
                        value="0-0",
                        placeholder="Example: 0,1-5,8,12-15 ~(default: first page)",
                        lines=1,
                        max_lines=1,
                    )
                    weasyprint_dll_directories_tb = gr.Textbox(
                        label="Path to weasyprint DLL libraries",
                        info='"C:\\Dat\\dev\\gtk3-runtime\\bin" or "C:\\msys64\\mingw64\\bin"',
                        placeholder="C:\\msys64\\mingw64\\bin",
                        lines=1,
                        max_lines=1,
                    )


        with gr.Accordion("🤗 HuggingFace Client Logout", open=True):  #, open=False):
            # Logout controls
            with gr.Row():
                #hf_login_logout_btn = gr.LoginButton(value="Sign in to HuggingFace 🤗", logout_value="Clear Session & Logout of HF: ({})", variant="huggingface")
                hf_login_logout_btn = gr.LoginButton(value="Sign in to HuggingFace 🤗", logout_value="Logout of HF: ({}) 🤗", variant="huggingface")
                #logout_btn = gr.Button("Logout from session & HF (inference) Client", variant="stop", )

            logout_status_md = gr.Markdown(visible=True)  #visible=False)
        
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

            config_load.file_types_list.extend(config_load.file_types_tuple)  ##allowed file types in global
            with gr.Column(elem_classes=["file-or-directory-area"]):
                with gr.Row():
                    file_btn = gr.UploadButton(
                    #file_btn = gr.File(
                        label="Upload Multiple Files",
                        file_count="multiple",
                        file_types= config_load.file_types_list,  #["file"],  ##config.file_types_list
                        #height=25,  #"sm",
                        size="sm",
                        elem_classes=["gradio-upload-btn"]
                    )
                    dir_btn = gr.UploadButton(
                    #dir_btn = gr.File(
                        label="Upload a Directory",
                        file_count="directory",
                        #file_types= config_load.file_types_list,   #["file"],  #Warning: The `file_types` parameter is ignored when `file_count` is 'directory'
                        ## [handled in accumulate_files] file_types - raised Error(gradio.exceptions.Error: "Invalid file type
                        #height=25,  #"0.5",
                        size="sm",
                        elem_classes=["gradio-upload-btn"]
                    )
            with gr.Accordion("Display uploaded", open=True):
                # Displays the accumulated file paths
                output_textbox = gr.Textbox(label="Accumulated Files", lines=3) #, max_lines=4)  #10
            
            with gr.Row():
                process_button = gr.Button("Process All Uploaded Files", variant="primary", interactive=False)
                clear_button = gr.Button("Clear All Uploads", variant="secondary", interactive=False)


        # --- PDF → Markdown tab ---
        with gr.Tab(" 📄 PDF ➜ Markdown (Flag for DEPRECATION)", interactive=False, visible=True):  #False
            gr.Markdown(f"#### {DESCRIPTION_PDF}")

            files_upload_pdf_fl = gr.File(
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
                interactive=True,  #False
                show_label=False,
            )

        # Initialise gr.State
        # The gr.State component to hold the accumulated list of files
        uploaded_file_list = gr.State([])   ##NB: initial value of `gr.State` must be able to be deepcopied
        uploaded_files_count = gr.State(0)   ## initial files count

        state_max_workers = gr.State(1)  #max_workers_sl,  #4
        state_max_retries = gr.State(2) #max_retries_sl,
        state_tz_hours    = gr.State(value=None)
        state_api_token   = gr.State(None)
        processed_file_state = gr.State([])   ##SMY: future: View and Download processed files


        def update_state_stored_value(new_component_input):
            """ Updates stored state: use for max_workers and max_retries """
            return new_component_input
        
        # Update gr.State values on slider components change. NB: initial value of `gr.State` must be able to be deepcopied
        max_workers_sl.change(update_state_stored_value, inputs=max_workers_sl, outputs=state_max_workers)
        max_retries_sl.change(update_state_stored_value, inputs=max_retries_sl, outputs=state_max_retries)
        tz_hours_num.change(update_state_stored_value, inputs=tz_hours_num, outputs=state_tz_hours)
        api_token_tb.change(update_state_stored_value, inputs=api_token_tb, outputs=state_api_token)
        

        # LLM Setting: Validate provider on change; warn but allow continue
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

        
        # HuggingFace Client Logout
        '''def get_login_token(state_api_token_arg, oauth_token: gr.OAuthToken | None=None):
            #oauth_token = get_token() if oauth_token is not None else state_api_token
            #oauth_token = oauth_token if oauth_token else state_api_token_arg
            if oauth_token:
                print(oauth_token)
                return oauth_token
            else:
                oauth_token = get_token()
                print(oauth_token)
                return oauth_token'''
        #'''
        def do_logout():    ##SMY: use with clear_state() as needed
            try:
                #ok = docextractor.client.logout()
                ok = docconverter.client.logout()
                # Reset token textbox on successful logout
                #msg = "✅ Logged out of HuggingFace and cleared tokens. Remember to log out of HuggingFace completely." if ok else "⚠️ Logout failed."
                msg = "✅ Session Cleared. Remember to close browser." if ok else "⚠️ HF client closing failed."
                
                return msg
                #return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session")
            except AttributeError:
                msg = "⚠️ HF client closing failed."
                
                return msg
                #return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session", interactive=False)
        #'''    
        def do_logout_hf():
            try:
                ok = docconverter.client.logout()
                # Reset token textbox on successful logout
                msg = "✅ Session Cleared. Remember to close browser." if ok else "⚠️ Logout & Session Cleared"
                #return gr.update(value=""), gr.update(visible=True, value=msg), gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value="Clear session", interactive=False)
                return msg
                #yield msg   ## generator for string
            except AttributeError:
                msg = "⚠️ Logout. No HF session"
                return msg
                #yield msg   ## generator for string
            
        #def custom_do_logout(hf_login_logout_btn_arg: gr.LoginButton, state_api_token_arg: gr.State):
        def custom_do_logout():
            #global state_api_token 
            '''  ##SMY: TO DELETE
            try:
                state_api_token_get= get_token() if "Clear Session & Logout of HF" in hf_login_logout_btn_arg.value else state_api_token_arg.value
            except AttributeError:
                #state_api_token_get= get_token() if "Clear Session & Logout of HF" in hf_login_logout_btn_arg else state_api_token_arg
                state_api_token_get = get_login_token(state_api_token_arg)
            '''
            #do_logout()
            #return gr.update(value="Sign in to HuggingFace 🤗")
            msg = do_logout_hf()
            ##debug
            #msg = "✅ Session Cleared. Remember to close browser." if "Clear Session & Logout of HF" in hf_login_logout_btn else "⚠️ Logout"  # & Session Cleared"
            return gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value=""), gr.update(visible=True, value=msg)  #, state_api_token_arg
            #yield gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value=""), gr.update(visible=True, value=msg)

        # Files, status, session clearing
        def clear_state():
            """
            Clears the accumulated state of uploaded file list, output textbox, files and directory upload.
            """
            #msg = f"Files list cleared: {do_logout()}"  ## use as needed
            msg = f"Files list cleared."
            #yield [], msg, '', ''
            #return [], f"Files list cleared.", [], []
            yield [], msg, None, None
            return [], 0, f"Files list cleared.", None, None

        #logout_btn.click(fn=clear_state, inputs=None, outputs=[uploaded_file_list, output_textbox, log_output, api_token_tb])
        hf_login_logout_btn.click(fn=custom_do_logout, inputs=None, outputs=[hf_login_logout_btn, api_token_tb, logout_status_md])  #, state_api_token])

        # --- PDF & HTML → Markdown tab ---
        # Event handler for the multiple file upload button
        file_btn.upload(
            fn=accumulate_files,
            inputs=[file_btn, uploaded_file_list],
            outputs=[uploaded_file_list, uploaded_files_count, output_textbox, process_button, clear_button]
        )

        # Event handler for the directory upload button
        dir_btn.upload(
            fn=accumulate_files,
            inputs=[dir_btn, uploaded_file_list],
            outputs=[uploaded_file_list, uploaded_files_count, output_textbox, process_button, clear_button]
        )

        # Event handler for the "Clear" button
        clear_button.click(
            fn=clear_state,
            inputs=None,
            outputs=[uploaded_file_list, output_textbox, file_btn, dir_btn],
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
            uploaded_files_count,  #files_count,  #pdf_files_count,
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
            api_token_tb,   #state_api_token,  #api_token_tb,
            openai_base_url_tb,
            openai_image_format_dd,
            state_max_workers, #gr.State(1),  #max_workers_sl,
            state_max_retries, #gr.State(2), #max_retries_sl,
            debug_cb,
            output_format_dd,
            output_dir_tb,
            use_llm_cb,
            force_ocr_cb,
            strip_existing_ocr_cb,
            disable_ocr_math_cb,
            page_range_tb,
            weasyprint_dll_directories_tb,
            tz_hours_num,   #state_tz_hours
            pooling_dd,
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
        ##TODO:
        #outputs=[uploaded_file_list, updated_files_count, output_textbox, process_button, clear_button]
        files_upload_pdf_fl.upload(
            fn=accumulate_files,
            inputs=[files_upload_pdf_fl, uploaded_file_list],
            outputs=[uploaded_file_list, uploaded_files_count, log_output, files_upload_pdf_fl, clear_button]
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
            inputs=[files_upload_pdf_fl],
            outputs=[files_count, log_output]
        )
        btn_html_count.click(
            fn=get_file_count,
            inputs=[files_upload_html],
            outputs=[html_files_count, log_output]
        )

    return demo

