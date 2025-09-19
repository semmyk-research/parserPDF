import os
from pathlib import Path
import sys
import ctypes
from typing import Union
from configparser import ConfigParser as config
from venv import logger
from utils.get_arg_name import get_arg_name_as_string
from utils.get_config import get_config_value
import traceback

from utils.logger import get_logger

logger = get_logger(__name__)

def set_weasyprint_library(libpath: Union[str, Path] = None, config_file: Union[str, Path] = "utils\\config.ini"):
    """ Loads Weasyprint backend dependency libraries to environment """
    # Check if the current platform is Windows
    if sys.platform == 'win32':
        
        #libgobject_path =  #"/path/to/your/custom/glib/install/lib/libgobject-2.0.so.0"
        if not libpath:
            '''cfg = config()
            cfg.read(config_file)  #"utils\\config.ini")
            lib_path = cfg["LIBRARIES_CAP"].get(f"WEASYPRINT_DLL_DIRECTORIES", "C:\\Dat\\dev\\gtk3-runtime\\bin")
            '''
            from file_handler.file_utils import find_file
            config_file = find_file("config.ini")  ##from file_handler.file_utils
            lib_path = get_config_value(config_file, "LIBRARIES_CAP", "WEASYPRINT_DLL_DIRECTORIES") if not libpath else "C:\\msys64\\mingw64\\bin"

            # Check if the file exists before attempting to load it
            #if not os.path.exists(libobject):
            if not Path(lib_path).exists():
                raise FileNotFoundError(f"The specified Weasyprint DLL Directory does not exist: {lib_path}. Follow Weasyprint installation guide or provide a valid GTK3-runtime path.")
            #logger.exception(f"gobject library path: {libgobject_path}")  ##debug
            
        try:
            # Set a new environment variable
            lib_path = lib_path  ##SMY: on dev machine, using extracted 'portable' GTK3 rather than installing 'MSYS2'
            os.environ["WEASYPRINT_DLL_DIRECTORIES"] = lib_path
            #logger.info(f"sets Weasyprint DLL library path: {lib_path}")   #debug
            
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception(f"Error setting environ: weasyprint backend dependency → {exc}\n{tb}", exc_info=True)  # Log the full traceback
            
            raise RuntimeWarning(f"✗ error during setting environ: weasyprint backend dependency → {exc}\n{tb}")
        

def load_library(libobject_name: Union[str, Path]):
    """ 
    Loads Weasyprint backend dependency libraries 
    usage:  list(map(load_library, library_list))  ##SMY: map the load_library function to each item in library_list
    The library list was starting to grow excessively, opt to setting environ   
    """
    # Check if the current platform is Windows
    if sys.platform == 'win32':
        
        #libgobject_path =  #"/path/to/your/custom/glib/install/lib/libgobject-2.0.so.0"
        cfg = config()
        cfg.read("utils\\config.ini")
        lib_path = cfg["libraries"].get(f"libobject_path", "C:\\Dat\\dev\\gtk3-runtime\\bin")
        lib_object_dll = get_arg_name_as_string(libobject_name)  ## future use

        # Construct the path to libgobject-2.0.dll
        #libgobject_path = os.path.join(os.environ.get('GLIB_PREFIX', 'C:\\glib'), 'bin', 'libgobject-2.0-0.dll')
        libobject = f"{lib_path}\\{libobject_name}.dll"   ##libgobject-2.0-0.dll"
        #print(f"Loading gobject library: {libgobject}")   #debug
        
        # Check if the file exists before attempting to load it
        #if not os.path.exists(libobject):
        if not Path(libobject).exists():
            raise FileNotFoundError(f"The specified library file does not exist: {libobject}")
        #print(f"gobject library path: {libgobject_path}")  ##debug

        # Load the library using ctypes
        try:
            ctypes_libgobject = ctypes.CDLL(libobject)
            #msg = f"libgobject-2.0-0.dll loaded successfully via ctypes. {str(ctypes_libgobject)}"
            #print(msg)  ##debug
        except OSError as exc:
            tb = traceback.format_exc()
            raise RuntimeWarning(f"Failed to load library: {exc}\n{tb}")  ##raise RuntimeError

## Test
#load_library("libpango-1.0-0")
#load_library("libgobject-2.0-0")


##SMY: Original implementation: TODO: for refactoring
def load_libgobject():
    # Check if the current platform is Windows
    if sys.platform == 'win32':
        
        #libgobject_path =  #"/path/to/your/custom/glib/install/lib/libgobject-2.0.so.0"
        cfg = config()
        cfg.read("utils\\config.ini")
        libgobject_path = cfg["libraries"].get("libgobject_path", "C:\\Dat\\dev\\gtk3-runtime\\bin")

        # Construct the path to libgobject-2.0.dll
        #libgobject_path = os.path.join(os.environ.get('GLIB_PREFIX', 'C:\\glib'), 'bin', 'libgobject-2.0-0.dll')
        libgobject = f"{libgobject_path}\\libgobject-2.0-0.dll"
        #print(f"Loading gobject library: {libgobject}")   #debug
        
        # Check if the file exists before attempting to load it
        if not os.path.exists(libgobject):
            raise FileNotFoundError(f"The specified library file does not exist: {libgobject}")
        #print(f"gobject library path: {libgobject_path}")  ##debug

        # Load the library using ctypes
        try:
            ctypes_libgobject = ctypes.CDLL(libgobject)
            #msg = f"libgobject-2.0-0.dll loaded successfully via ctypes. {str(ctypes_libgobject)}"
            #print(msg)  ##debug

            return ctypes_libgobject
        except OSError as exc:
            tb = traceback.format_exc()
            raise RuntimeWarning(f"Failed to load library: {exc}\n{tb}")  ##raise RuntimeError


    # Load the library using ctypes (Linux/macOS)
    # Construct the path to libgobject-2.0.so.0 in the custom GLib installation
    #libgobject_path = os.path.join(os.environ.get('GLIB_PREFIX', '/opt/glib'), 'lib', 'libgobject-2.0.so.0') 
    #print("This script is intended to run on Unix-like systems, not Windows.")
    else:
        # Load the library using ctypes (Linux/macOS)
        # Construct the path to libgobject-2.0.so.0 in the custom GLib installation
        libgobject_path = os.path.join(os.environ.get('GLIB_PREFIX', '/opt/glib'), 'lib', 'libgobject-2.0.so.0') 
        #print("This script is intended to run on Unix-like systems, not Windows.")

        return libgobject_path
