
from __future__ import annotations

from typing import Optional   #Iterable, Literal
#import os
#import time
import traceback
#from huggingface_hub import InferenceClient, login, logout as hf_logout

from llm.llm_login import login_huggingface, is_login_huggingface

import dotenv
#dotenv.load_dotenv(".env")


from utils.logger import get_logger

## Get logger instance
logger = get_logger(__name__)


class OpenAIChatClient:
    """
    Provider‐agnostic OpenAI-based LLM client interface.
    Compatible with `huggingface_hub.InferenceClient` setup and chat calls.

    - base_url="https://router.huggingface.co/v1",
    - api_key=os.environ["HF_TOKEN"],
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 hf_provider: Optional[str] = None,
                 base_url: Optional[str] = "https://router.huggingface.co/v1",  #None,
                 api_token: Optional[str] = None,
                 temperature: Optional[float] = 0.2,
                 top_p: Optional[float] = 0.2,
                ) -> None:
        
        try:
            openai_api_key_env = dotenv.get_key(".env", "OPENAI_API_KEY")
            self.model_id = f"{model_id}:{hf_provider}" if hf_provider is not None else model_id  ##concatenate so HF can pipe to Hf provider
            self.hf_provider = hf_provider
            self.base_url = base_url  #"https://router.huggingface.co/v1"  #%22"  #HF API proxy
            #self.token = api_token if api_token else None   ##debug
            self.token = openai_api_key_env if openai_api_key_env else api_token  #dotenv.get_key(".env", "OPENAI_API_KEY")
            #self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")  ## not preferred
            login_huggingface(self.token) if not is_login_huggingface() else logger.log(level=20, msg=f"You are logged in to HF Hub already") ## attempt login if not already logged in. NB: HF CLI login prompt would not display in Process Worker.
            #self.fake_token = api_token or "a1b2c3" #or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            self.openai_api_key = self.token  #self.fake_token
            self.temperature = temperature
            self.top_p = top_p
            self.islogged_in = is_login_huggingface()

            logger.log(level=2, msg="initialised OpenAIChatClient:", extra={"base_url": self.base_url, "openai_api_key": self.openai_api_key})
            
        except Exception as exc:
            #logger.error(f"OpenAI client_init_failed", extra={"error": str(exc)}")
            tb = traceback.format_exc()
            logger.exception(f'✗ OpenAI client_init_failed", extra={"error": str(exc)}\n{tb}', exc_info=True)
            raise RuntimeError(f"✗ Failed to initialise OpenAI client: {exc}\n{tb}")
      
        #login_huggingface(self.token) if not is_login_huggingface() else logger.log(level=20, msg=f"logged in to HF Hub already") ## attempt login if not already logged in. NB: HF CLI login prompt would not display in Process Worker.

####IN PROGRESS
# 
"""
## HuggingFace API-proxy Inference Provider - https://huggingface.co/docs/inference-providers/index?python-clients=openai
## https://huggingface.co/openai/gpt-oss-20b?inference_api=true&inference_provider=fireworks-ai&language=python&client=openai

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

stream = client.chat.completions.create(
    model="openai/gpt-oss-20b:fireworks-ai",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
"""
