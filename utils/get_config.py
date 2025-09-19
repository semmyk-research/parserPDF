from configparser import ConfigParser as config
from typing import Union
from pathlib import Path
#from utils.get_arg_name import get_arg_name_as_string
import traceback

'''
##debug
import sys
from pathlib import Path
#base_grandparent = Path(__file__).resolve().parent.parent
grandparent_dir = Path('.').resolve() #.parent.parent
sys.path.insert(0, f"{grandparent_dir}")  #\\file_handler")
##end debug
#'''
#import file_handler
from file_handler.file_utils import find_file

def get_config_value(config_file:Path, section_key:str, parameter:str, fallback:str=None) -> str:   # configfile: Union[str, Path]="utils\\config.ini"):
    """ Load config file, locate section, read parameter and return value 
    
    Args:
        section_key: The section key
        parameter: The parameter key to read from the configuration file
        fallback: The fallback parameter if the parameter value not found
        config_file: The configuration file to load.
    
    Returns:
        The key parameter value.
    
    Raises:
        RuntimeWarning: If the configuration file cannot be loaded or parameter key found.
    """
    
    try:
        #config_file = find_config(config_file)
        cfg = config()
        if config_file.is_file():
            cfg.read(config_file)
            param_value = cfg[section_key].get(option=parameter, fallback=fallback)  #"C:\\Dat\\dev\\gtk3-runtime\\bin")
            return param_value
        else:
            raise RuntimeWarning(f"Configuration file not found: {config_file}")
    except KeyError as exc:
        tb = traceback.format_exc()
        raise RuntimeWarning(f"Error loading parameter key: {exc}\n{tb}")
    except Exception as exc:
        tb = traceback.format_exc()
        raise RuntimeWarning(f"Error loading config or parameter key: {exc}\n{tb}")
        #pass



##debug
'''
config_file_path = find_file("config.ini")  #file_handler.file_utils.
config_value = get_config_value(config_file_path, "LIBRARIES_CAP", "WEASYPRINT_DLL_DIRECTORIES")
print(f"config value: {config_value}")
'''

##SMY: moved to file_handler.file_utils as find_file()
def find_config(config_file_name: str = "config.ini") -> config:  #configparser.ConfigParser:
    """
    Finds and loads a configuration file named 'config_file_name' from the
    same directory or a parent directory of the calling script.
    
    Args:
        config_file_name: The name of the configuration file to load.
    
    Returns:
        A ConfigParser object with the loaded configuration.
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    # Start the search from the directory of the file this function is in
    search_path = Path(__file__).resolve().parent

    # Walk up the directory tree until the config file is found
    for parent in [search_path, *search_path.parents]:
        config_path = parent / config_file_name
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(f"Configuration file '{config_file_name}' not found.")



def get_config_value_old(section:str, parameter:str, fallback:str=None, configfile: Union[str, Path]="utils\\config.ini"):
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

##TODO:  //STOP
# ##SMY: HF Space RuntimeWarning: Error loading config: 'MARKER_CAP'
'''
from pathlib import Path
import configparser
from typing import Optional

def load_config(config_file_name: str = "config.ini") -> configparser.ConfigParser:
    """
    Finds and loads a configuration file named 'config_file_name' from the
    same directory or a parent directory of the calling script.
    
    Args:
        config_file_name: The name of the configuration file to load.
    
    Returns:
        A ConfigParser object with the loaded configuration.
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    # Start the search from the directory of the file this function is in
    search_path = Path(__file__).resolve().parent

    # Walk up the directory tree until the config file is found
    for parent in [search_path, *search_path.parents]:
        config_path = parent / config_file_name
        if config_path.is_file():
            config = configparser.ConfigParser()
            config.read(config_path)
            return config

    raise FileNotFoundError(f"Configuration file '{config_file_name}' not found.")
'''
