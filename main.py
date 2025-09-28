# main.py
import os
from pathlib import Path

from ui.gradio_ui import build_interface
from utils.logger import get_logger, setup_logging

setup_logging()  ## set logging
#logger = get_logger("pypdfmd")
logger = get_logger("parserpdf")

if __name__ == "__main__":
    # Ensure the working directory is clean
    #os.chdir(os.path.dirname(__file__))
    ## script working dir absolute path
    script_dir = Path(__file__).resolve().parent
    ## change the cwd to the script's dir
    os.chdir(script_dir)    ##Path.cwd()

    demo = build_interface()
    #demo.launch(debug=True, show_error=True ,ssr_mode=True)  #(share=True)  # share=True for public link; remove in production
    demo.launch(debug=True, show_error=True, ssr_mode=False)
