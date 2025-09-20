from huggingface_hub import HfApi, login, logout, get_token
import os
import traceback
from time import sleep
from typing import Optional

from utils.logger import get_logger

## Get logger instance
logger = get_logger(__name__)

def login_huggingface(token: Optional[str] = None):
    """
    Login to Hugging Face account. Prioritize CLI login for privacy and determinism.

    Attempts to log in to Hugging Face Hub.
    First, it tries to log in interactively via the Hugging Face CLI.
    If that fails, it falls back to using a token provided as an argument or
    found in the environment variables HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.

    If both methods fail, it logs a warning and continues without logging in.
    """

    logger.info("Attempting Hugging Face login...")
        
    # Disable implicit token propagation for determinism
    # Explicitly disable implicit token propagation; we rely on explicit auth or env var
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    token = token
    # Privacy-first login: try interactive CLI first; fallback to provided/env token only if needed
    try:
        if HfApi.whoami():
            logger.info("✔️ hf_login already", extra={"mode": "cli"})
            #return True
        else:
            login()
            sleep(5)  ##SMY pause for login. Helpful: pool async opex 
            logger.info("✔️ hf_login already", extra={"mode": "cli"})
            #return True
    except Exception as exc:
        # Respect common env var names; prefer explicit token arg when provided
        fallback_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
        if fallback_token:
            try:
                login(token=fallback_token)
                token = fallback_token
                logger.info("✔️ hf_login through fallback", extra={"mode": "token"})  ##SMY: This only displays if token is provided
            except Exception as exc_token:
                logger.warning("❌ hf_login_failed", extra={"error": str(exc_token)})
        else:
            logger.warning("❌ hf_login_failed", extra={"error": str(exc)})
            # Silent fallback; client will still work if token is passed directly
            #pass

def is_login_huggingface():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        HfApi().whoami()
        logger.log(level=20, msg=("✔️ You are logged in."), extra={"is_logged_in": True})
        return True
    except HfHubHTTPError as exc:
        # A 401 status code indicates an authentication error.
        if exc.response.status_code == 401:
            print("⚠️ You are not logged in. You can still access public models.")
        else:
            # Handle other HTTP errors if necessary
            #print(f"An unexpected HTTP error occurred: {exc}")
            tb = traceback.format_exc()
            logger.exception(f"✗ An unexpected HTTP error occurred: → {exc}\n{tb}", exc_info=True)
            #raise RuntimeError(f"✗ An unexpected HTTP error occurred: → {exc}\n{tb}") from exc
            return False
        