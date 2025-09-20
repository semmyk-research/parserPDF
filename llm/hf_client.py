from __future__ import annotations

from typing import Iterable, Literal, Optional
import os
import time
import traceback
from huggingface_hub import InferenceClient, login, logout as hf_logout

from llm.llm_login import login_huggingface, is_login_huggingface

from utils.logger import get_logger

## Get logger instance
logger = get_logger(__name__)


class HFChatClient:
    """
    Provider‐agnostic LLM client interface.
    Encapsulate `huggingface_hub.InferenceClient` setup and chat calls.

    Backends:
    - model: plain HF model id (e.g., "HuggingFaceH4/zephyr-7b-beta")
    - provider: provider-routed id (e.g., "openai/gpt-oss-120b:fireworks-ai")
    - endpoint: full inference endpoint URL (e.g., "http://localhost:1234").
    """

    def __init__(self,
        #api_token: str,
        #model_id: str = "gpt2",
        provider: str = "huggingface",  ## "huggingface2", "openai"
        model_id: str = "openai/gpt-oss-120b", ##default_model
        hf_provider: str = "huggingface",
        endpoint_url: Optional[str] = None,
        #backend: Literal["model", "provider", "endpoint"] = [],
        backend_choice: Optional[str] = None,  #choices=["model-id", "provider", "endpoint"]
        system_message: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.1,
        stream: bool = False,
        api_token: Optional[str] = None
        ) -> None:
        
        try:
            self.model_id = model_id
            self.provider = provider.lower()
            self.hf_provider = hf_provider.lower()
            self.endpoint_url = endpoint_url
            #self.backend = backend
            #self.backend_literal: Literal["model", "provider", "endpoint"] = (
            '''
            self.backend: Literal["model", "provider", "endpoint"] = (
                "model" if backend_choice == "Hugging Face Model ID" else (
                    "provider" if backend_choice == "HF Provider Route" else "endpoint")
                ),
            '''
            self.backend: Literal["model", "provider", "endpoint"] = (
                "model" if backend_choice == "model-id" else (
                    "provider" if backend_choice == "provider" else "endpoint")
                )  ## see Gradio backend_choice dropdown
            self.system_message = system_message
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.stream = stream
            self.token = api_token if api_token else None   #""  # invalid; preserved
            #self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")  ## not preferred

            self.base_url = "https://router.huggingface.co/v1"  #%22"  #HF API proxy
        except Exception as exc:
            #logger.error(f"client_init_failed", extra={"error": str(exc)}")
            tb = traceback.format_exc()
            logger.exception(f'✗ client_init_failed", extra={"error": str(exc)}\n{tb}', exc_info=True)
            raise RuntimeError(f"✗ Failed to initialise client: {exc}\n{tb}")

        ##SMY: //TOBE: Deprecated : Moved to llm.llm_login
        '''
        # # Disable implicit token propagation for determinism
        # Explicitly disable implicit token propagation; we rely on explicit auth or env var
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        
        # Privacy-first login: try interactive CLI first; fallback to provided/env token only if needed
        try:
            login()
            time.sleep(15)  ##SMY pause for login. Helpful: pool async opex 
            logger.info("hf_login", extra={"mode": "cli"})
        except Exception as exc:
            # Respect common env var names; prefer explicit token arg when provided
            fallback_token = self.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if fallback_token:
                try:
                    login(token=fallback_token)
                    self.token = fallback_token
                    logger.info("hf_login", extra={"mode": "token"})
                except Exception as exc_token:
                    logger.warning("hf_login_failed", extra={"error": str(exc_token)})
            else:
                logger.warning("hf_login_failed", extra={"error": str(exc)})
                # Silent fallback; client will still work if token is passed directly
                #pass
        '''
        
        login_huggingface(self.token) if not is_login_huggingface() else logger.log(level=20, msg=f"You are logged in to HF Hub already") ## attempt login if not already logged in. NB: HF CLI login prompt would not display in Process Worker.
        ##SMY: TODO: Mapped with openai_client.py
        #self.islogged_in = is_login_huggingface()

    @staticmethod
    def _normalise_history(history: list, system_message: str, latest_user_message: str) -> list[dict]:
        """ 
        `prompt` prefixed by system_message if set
        Normalise chat history to list of {"role": role, "content": content} dicts.
        Supports both dict and tuple formats for history items.
        """
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        for item in history or []:
            if isinstance(item, dict) and "role" in item and "content" in item:
                if item["role"] in ("user", "assistant"):
                    messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                usr, asst = item
                if usr:
                    messages.append({"role": "user", "content": usr})
                if asst:
                    messages.append({"role": "assistant", "content": asst})
        messages.append({"role": "user", "content": latest_user_message})
        return messages

    @staticmethod
    def _initialise_client(self,
        backend: Literal["model", "provider", "endpoint"], 
        model_id: Optional[str] = None, 
        hf_provider: Optional[str] = None, 
        endpoint_url: Optional[str] = None, 
        token: Optional[str] = None) -> InferenceClient:

        try:
            match backend:
                case "endpoint" | "model":
                    logger.debug("_initialise_client: initialising with:", extra={"model":model_id})  ## debug
                    hf_client = InferenceClient(model=model_id or endpoint_url, token=token)   #endpoint=target)   ##, token=api_token or self.token)
                    logger.log(20, "client: ", extra={"model":model_id})  ## debug
                case "provider":
                    logger.info("_initialise_client: initialising with:", extra={"provider":hf_provider})  ## debug
                    hf_client = InferenceClient(provider=hf_provider, model=model_id, token=token)  ##, token=api_token or self.token)
                    #client = client(model = model_id, provider=provider, token=token)   ##target
                    logger.log(20, "client: ", extra={"backend":backend})  ## debug
                case _:
                    raise ValueError("Invalid backend.")
            return hf_client
        except Exception as exc:
            logger.log(40, "_initialise_client: client_init_failed", extra={"error": str(exc)})  ## debug
            raise RuntimeError(f"_initialise_client: Failed to initialise client: {exc}")

    ## wrap HF client for marker
    def chat_fn(
        self,
        message: str,
        history: list = [],
        ) -> Iterable[str]:
        """    
        messages = self._normalise_history(history, system_message, message)
        token = api_token or self.token
        """
        ## set prompt and token
        messages = self._normalise_history(message, history, self.system_message)
        #token = api_token or self.token
        #token = self.token  ## redundant

        logger.log(20,"chat: initialising client", extra={
            "backend": self.backend, "model": self.model_id, "provider": self.hf_provider, "endpoint": self.endpoint_url,
            "stream": self.stream, "max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p,
        })

        ## initialised client
        try:
            client = self._initialise_client(self, self.backend, self.model_id, self.hf_provider, self.endpoint_url, self.token)  #api_token)
            logger.log(20, "chat: client initialised")  ## debug
        except Exception as exc:
            ##logger.error
            logger.log(40,"chat client_init_failed", extra={"error": str(exc)})
            raise RuntimeError(f"chat: Failed to initialise client: {exc}")
        
        logger.log(20, "chat_start", extra={
                "backend": self.backend, "model": self.model_id, "provider": self.hf_provider, "endpoint": self.endpoint_url,
            "stream": self.stream, "max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p,
                })
        
        if self.stream:
            acc = ""
            for chunk in client.chat_completion(
                messages=messages,
                #model=client.model,  ## moved back to client initialise
                max_tokens=self.max_tokens,
                stream=True,
                temperature=self.temperature,
                top_p=self.top_p,
            ):
                delta = getattr(chunk.choices[0].delta, "content", None) or ""
                if delta:
                    acc += delta
                    yield acc
            return

        result = client.chat_completion(
            messages=messages,
            #model=client.model,  ## moved back to client initialised
            max_tokens=self.max_tokens,
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            )
        yield result.choices[0].message.content

        '''
        ## future consideration
        response = client.text_generation(
            #model=model_name,
            inputs=prompt,
            parameters={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        return response[0].generated_text
        '''

    def logout(self) -> bool:
        """Logout from Hugging Face and clear in-process tokens.

        Returns True on success, False otherwise.
        """
        try:
            hf_logout()
        except Exception as exc:
            logger.error("hf_logout_failed", extra={"error": str(exc)})
            return False
        # Clear process environment tokens
        for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            if key in os.environ:
                os.environ.pop(key, None)
        self.token = None
        logger.info("hf_logout_success")
        return True

