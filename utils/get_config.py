from configparser import ConfigParser as config
from typing import Union
from pathlib import Path
#from utils.get_arg_name import get_arg_name_as_string
import traceback

def get_config_value(section:str, parameter:str, fallback:str=None, configfile: Union[str, Path]="utils\\config.ini"):
    """ Load config file, locate section, read parameter and return value """
    
    try:
        cfg = config()
        cfg.read(configfile)
        param_value = cfg[section].get(option=parameter, fallback=fallback)  #"C:\\Dat\\dev\\gtk3-runtime\\bin")
        return param_value
    except Exception as exc:
        tb = traceback.format_exc()
        raise RuntimeWarning(f"Error loading config: {exc}\n{tb}")
        #pass