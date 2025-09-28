import os
from pathlib import Path
import traceback
#import time
from typing import Dict, Any, Type, Optional, Union #, BaseModel
from pydantic import BaseModel

from marker.models import create_model_dict
#from marker.converters.extraction import ExtractionConverter as MarkerExtractor  ## structured pydantic extraction
from marker.converters.pdf import PdfConverter as MarkerConverter  ## full document convertion/extraction
from marker.config.parser import ConfigParser  ## Process custom configuration
from marker.services.openai import OpenAIService as MarkerOpenAIService
#from sympy import Union

#from llm.hf_client import HFChatClient
from llm.openai_client import OpenAIChatClient
from file_handler.file_utils import collect_pdf_paths, collect_html_paths, collect_markdown_paths, create_outputdir
from utils.lib_loader import load_library

from utils.logger import get_logger

logger = get_logger(__name__)

# Full document converter
class DocumentConverter:
    """ 
    Business logic wrapper using Marker OpenAI LLM Services to
    convert documents (PDF, HTML files) into markdowns + assets. 
    """

    def __init__(self,
        #provider: str,
        model_id: str,
        #base_url: str,
        hf_provider: str,
        #endpoint_url: str,
        #backend_choice: str,
        #system_message: str,
        #max_tokens: int,
        temperature: float,
        top_p: float,
        #stream: bool,
        api_token: str,
        openai_base_url: str = "https://router.huggingface.co/v1",
        openai_image_format: Optional[str] = "webp",
        #max_workers: Optional[str] = 4,
        max_retries: Optional[int] = 2,
        output_format: str = "markdown",
        output_dir: Optional[Union[str, Path]] = "output_dir",
        use_llm: Optional[bool] = None,  #bool = False,  #Optional[bool] = False,  #True,
        page_range: Optional[str] = None,  #str = None  #Optional[str] = None,  
        ):

        #self.converter = None  #MarkerConverter
        self.model_id = model_id  #"model_name"
        self.openai_api_key = api_token  ## to replace dependency on self.client.openai_api_key
        self.openai_base_url = openai_base_url  #,  #self.base_url,
        self.temperature = temperature   #, self.client.temperature,
        self.top_p = top_p               # self.client.top_p,
        self.llm_service = MarkerOpenAIService
        self.openai_image_format = openai_image_format  #"png"  #better compatibility
        self.max_retries = max_retries  ## pass to __call__
        self.output_dir = output_dir
        self.use_llm = use_llm[0] if isinstance(use_llm, tuple) else use_llm,  #False,  #True,
        #self.page_range = page_range[0] if isinstance(page_range, tuple) else page_range   ##SMY: iterating twice because self.page casting as hint type tuple!
        self.page_range = page_range if page_range else None
        # self.page_range = page_range[0] if isinstance(page_range, tuple) else page_range if isinstance(page_range, str) else None,  ##Example: "0,4-8,16"  ##Marker parses as List[int]  #]debug  #len(pdf_file)
        '''
        if isinstance(page_range, tuple | str):
            self.page_range = page_range[0] if isinstance(page_range, tuple) else page_range
        else:
            self.page_range = None
        '''

        # 0) Instantiate the LLM Client (OPENAIChatClient): Get a provider‐agnostic chat function
        ##SMY: #future. Plan to integrate into Marker: uses its own LLM services (clients). As at 1.9.2, there's no huggingface client service.
        try:
            self.client = OpenAIChatClient(
            model_id=model_id,
            hf_provider=hf_provider,
            #base_url=base_url,
            api_token=api_token,
            temperature=temperature,
            top_p=top_p,
            )
            logger.log(level=20, msg="✔️ OpenAIChatClient instantiated:", extra={"model_id": self.client.model_id, "chatclient": str(self.client)})

        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error initialising OpenAIChatClient: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error initialising OpenAIChatClient: {exc}\n{tb}")  #.with_traceback(tb)

        # 1) # Define the custom configuration for the Hugging Face LLM.
                # Use typing.Dict and typing.Any for flexible dictionary type hints 
        try:
            self.config_dict: Dict[str, Any] = self.get_config_dict(model_id=model_id, llm_service=str(self.llm_service), output_format=output_format)
            #self.config_dict.pop("page_range") if self.config_dict.get("page_range")[0] is None else None  ##SMY: execute if page_range is none. `else None` ensures valid syntactic expression

            ##SMY: if falsely empty tuple () or None, pop the "page_range" key-value pair, else do nothing if truthy tuple value (i.e. keep as-is)
            self.config_dict.pop("page_range", None) if not self.config_dict.get("page_range") else None

            logger.log(level=20, msg="✔️ config_dict custom configured:", extra={"service": "openai"})  #, "config": str(self.config_dict)})

        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error configuring custom config_dict: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error configuring custom config_dict: {exc}\n{tb}")  #.with_traceback(tb)

        # 2) Use the Marker's ConfigParser to process configuration.
            # The `ConfigParser` class is explicitly imported and used as the type hint.
        try:
            config_parser: ConfigParser = ConfigParser(self.config_dict)
            logger.log(level=20, msg="✔️ parsed/processed custom config_dict:", extra={"config": str(config_parser)})  #.config_dict)})

        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error parsing/processing custom config_dict: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error parsing/processing custom config_dict: {exc}\n{tb}")  #.with_traceback(tb)
        
        # 3) Create the artifact dictionary and retrieve the LLM service.
        try:
            #self.artifact_dict: Dict[str, Any] = self.get_create_model_dict  ##SMY: Might have to eliminate function afterall
            self.artifact_dict: Dict[str, Type[BaseModel]] = create_model_dict()  ##SMY: BaseModel for Any??
            #logger.log(level=20, msg="✔️ Create artifact_dict and llm_service retrieved:", extra={"llm_service": self.llm_service})
        
        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error creating artifact_dict or retrieving LLM service: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error creating artifact_dict or retrieving LLM service: {exc}\n{tb}")  #.with_traceback(tb)

        # 4) Instantiate Marker's MarkerConverter (PdfConverter) with config managed by config_parser
        try:
            llm_service_str = str(self.llm_service).split("'")[1]  ## SMY: split and slicing  ##Gets the string value

            # sets api_key required by Marker 
            os.environ["OPENAI_API_KEY"] = api_token if api_token !='' or None else self.openai_api_key  ## to handle Marker's assertion test on OpenAI
            logger.log(level=20, msg="self.converter: instantiating MarkerConverter:", extra={"llm_service_str": llm_service_str, "api_token": api_token})  ##debug
            
            #self.converter: MarkerConverter = MarkerConverter(
            self.converter = MarkerConverter(
                #artifact_dict=self.artifact_dict,
                artifact_dict=create_model_dict(),
                config=config_parser.generate_config_dict(),
                #llm_service=self.llm_service  ##SMY expecting str but self.llm_service, is service object marker.services of type BaseServices
                llm_service=llm_service_str    ##resolve
            )
            
            logger.log(level=20, msg="✔️ MarkerConverter instantiated successfully:", extra={"converter.config": str(self.converter.config.get("openai_base_url")), "use_llm":self.converter.use_llm})
            #return self.converter  ##SMY: to query why did I comment out?. Bingo: "__init__() should return None, not 'PdfConverter'"
        except Exception as exc:
            tb = traceback.format_exc
            logger.exception(f"✗ Error initialising MarkerExtractor: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error initialising MarkerExtractor: {exc}\n{tb}")
        
        # Define the custom configuration for HF LLM.
    def get_config_dict(self, model_id: str, llm_service=MarkerOpenAIService, output_format: Optional[str] = "markdown" ) -> Dict[str, Any]:
        """ Define the custom configuration for the Hugging Face LLM. """

        try:
            ## Enable higher quality processing with LLMs.  ## See MarkerOpenAIService,  
            #llm_service = llm_service.removeprefix("<class '").removesuffix("'>")  # e.g <class 'marker.services.openai.OpenAIService'>
            llm_service  = str(llm_service).split("'")[1]  ## SMY: split and slicing
            self.use_llm = self.use_llm[0]
            self.page_range = self.page_range[0] if isinstance(self.page_range, tuple) else self.page_range #if isinstance(self.page_range, str) else None,  ##SMY: passing as hint type tuple!
            

            config_dict = {
                "output_format" : output_format,     #"markdown",
                "openai_model"   : self.model_id,    #self.client.model_id,  #"model_name"
                "openai_api_key" : self.client.openai_api_key,   #self.client.openai_api_key,  #self.api_token,
                "openai_base_url": self.openai_base_url,  #self.client.base_url,  #self.base_url,
                "temperature"    : self.temperature,      #self.client.temperature,
                "top_p"          : self.top_p,            #self.client.top_p,
                "openai_image_format": self.openai_image_format, #"webp",  #"png"  #better compatibility
                "max_retries"    : self.max_retries,  #3,  ## pass to __call__
                "output_dir"     : self.output_dir,
                "use_llm"        : self.use_llm,      #False,  #True,
                "page_range"     : self.page_range,   #]debug  #len(pdf_file)
            }
            return config_dict
        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error configuring custom config_dict: {exc}\n{tb}")
            raise RuntimeError(f"✗ Error configuring custom config_dict: {exc}\n{tb}")  #").with_traceback(tb)
            #raise

    ##SMY: flagged for deprecation
    ##SMY: marker prefer default artifact dictionary (marker.models.create_model_dict) instead of overridding
    #def get_extraction_converter(self, chat_fn):
    def get_create_model_dict(self):
        """
        Wraps the LLM chat_fn into marker’s artifact_dict
        and returns an ExtractionConverter for PDFs & HTML.
        """
        return create_model_dict() 
        #artifact_dict = create_model_dict(inhouse_chat_model=chat_fn)      
        #return artifact_dict            

## SMY: Kept for future implementation (and historic reasoning). Keeping the classes separate to avoid confusion with the original implementation
'''
class DocumentExtractor:
    """ 
    Business logic wrapper using HFChatClient and Marker to
    convert documents (PDF, HTML files) into markdowns + assets
    Wrapper around the Marker extraction converter for PDFs & HTML. 
    """

    def __init__(self,
        provider: str,
        model_id: str,
        hf_provider: str,
        endpoint_url: str,
        backend_choice: str,
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        api_token: str,
        ):
        # 1) Instantiate the LLM Client (HFChatClient): Get a provider‐agnostic chat function
        try:
            self.client = HFChatClient(
            provider=provider,    
            model_id=model_id,
            hf_provider=hf_provider,
            endpoint_url=endpoint_url,
            backend_choice=backend_choice,       #choices=["model-id", "provider", "endpoint"]
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            api_token=api_token,
            )
            logger.log(level=20, msg="✔️ HFChatClient instantiated:", extra={"model_id": model_id, "chatclient": str(self.client)})

        except Exception as exc:
            tb = traceback.format_exc()   #exc.__traceback__
            logger.exception(f"✗ Error initialising HFChatClient: {exc}")
            raise RuntimeError(f"✗ Error initialising HFChatClient: {exc}").with_traceback(tb)
            #raise

        # 2) Build Marker's artifact dict using the client's chat method
        self.artifact_dict = self.get_extraction_converter(self.client)
        
        # 3) Instantiate Marker's ExtractionConverter (ExtractionConverter)
        try:
            self.extractor = MarkerExtractor(artifact_dict=self.artifact_dict)
        except Exception as exc:
            logger.exception(f"✗ Error initialising MarkerExtractor: {exc}")
            raise RuntimeError(f"✗ Error initialising MarkerExtractor: {exc}")
    
    ##SMY: marker prefer default artifact dictionary (marker.models.create_model_dict) instead of overridding
    def get_extraction_converter(self, chat_fn):
        """
        Wraps the LLM chat_fn into marker’s artifact_dict
        and returns an ExtractionConverter for PDFs & HTML.
        """
        
        artifact_dict = create_model_dict(inhouse_chat_model=chat_fn)
        return artifact_dict
'''

