# file_handler/file_utils.py
#import os
from pathlib import Path
from itertools import chain
from typing import List, Union, Any, Mapping
from PIL import Image

import utils.config as config

##SMY: Might be deprecated vis duplicated. See marker/marker/config/parser.py  ~ https://github.com/datalab-to/marker/blob/master/marker/config/parser.py#L169
#def create_outputdir(root: Union[str, Path], out_dir:Union[str, Path] = None) -> Path:  #List[Path]:
def create_outputdir(root: Union[str, Path], output_dir_string:str = None) -> Path:  #List[Path]:
    """ Create output dir under the input folder """
    
    '''  ##preserved for future implementation if needed again
    root = root if isinstance(root, Path) else Path(root)  
    #root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist: cannot create output dir.")
    out_dir = out_dir if out_dir else "output_md"  ## SMY: default to outputdir in config file = "output_md"
    output_dir = root.parent / out_dir  #"md_output"  ##SMY: concatenating output str with src Path
    '''

    ## map to img_path. Opt to putting output within same output_md folder rather than individual source folders
    output_dir_string = output_dir_string if output_dir_string else "output_dir"  ##redundant SMY: default to outputdir in config file = "output_md"
    output_dir = Path("data") / output_dir_string  #"output_md"  ##SMY: concatenating output str with src Path
    output_dir.mkdir(mode=0o2644, parents=True, exist_ok=True)
    return output_dir

def check_create_file(filename: str, dir_path: Union[str, Path]="logs") -> Path:
    """
    check if File exists, else create one ad return the file path.

    Args:
        directory_path (str): The path to the directory.
        filename (str): The name of the file to check/create.
    Returns:
        The pathlib.Path object for the file
    """
    
    #file_dir = Path("logs") / file_dir if not isinstance(file_dir, Path) else Path(file_dir)
    dir_path = dir_path if isinstance(dir_path, Path) else Path(dir_path)

    # Ensure the directory exists
    # Create the parent directory if it doesn't exist.
    # `parents=True` creates any missing parent directories.
    # `exist_ok=True` prevents an error if the directory already exists.
    dir_path.mkdir(parents=True, exist_ok=True, mode=0o2664)  #, mode=0o2644)
    dir_path.chmod(0) 
    
    file_path = dir_path / filename  # Concatenate directory and filename to get full path
    print(f"file_path:  {file_path}")  ##]debug
    
    file_path.touch(exist_ok=True, mode=0o2664)  # Creates an empty file if it doesn't exists

    '''
    if not file_path.exists():       # check if file doesn't exist
        file_path.touch(exist_ok=True, mode=0o2664)  # Creates an empty file if it doesn't exists
        #file_dir.touch(mode=0o2644, exist_ok=True)  #, parents=True)  ##SMY: Note Permission Errno13 - https://stackoverflow.com/a/57454275
        #file_dir.chmod(0)
    ''' 
    
    return file_path

## debug
#print(f'file: {check_create_file("app_logging.log")}')

def is_file_with_extension(path_obj: Path) -> bool:
    """
    Checks if a pathlib.Path object is a file and has a non-empty extension.
    """
    path_obj = path_obj if isinstance(path_obj, Path) else Path(path_obj) if isinstance(path_obj, str) else None
    return path_obj.is_file() and bool(path_obj.suffix)

def process_dicts_data(data:Union[dict, list[dict]]):
    """ Returns formatted JSON string for a single dictionary or a list of dictionaries"""
    import json
    from pathlib import WindowsPath
    #from typing import dict, list

    # Serialise WindowsPath objects to strings using custom json.JSoNEncoder subclass
    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, WindowsPath):
                return str(obj)
            # Let the base class default method raise the TypeError for other types
            return json.JSONEncoder.default(self, obj)

    # Convert the list of dicts to a formatted JSON string
    formatted_string = json.dumps(data, indent=4, cls=PathEncoder)
    
    return formatted_string

##NB: Python =>3.10, X | Y equiv to the type checker as Union[X, Y]
def collect_pdf_html_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    patterns = ["*.pdf", "*.html"]  #, "*.htm*"]
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    #pdfs_htmls = [p for p in root.rglob("*.pdf", "*.html", "*.htm*") if p.is_file()]
    #pdfs_htmls = [chain.from_iterable(root.rglob(pattern) for pattern in patterns)]
    # Use itertools.chain to combine the generators from multiple rglob calls
    pdfs_htmls = list(chain.from_iterable(root.rglob(pattern) for pattern in patterns))

    return pdfs_htmls

def collect_pdf_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    pdfs = [p for p in root.rglob("*.pdf") if p.is_file()]
    return pdfs

def collect_html_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    htmls = [p for p in root.rglob("*.html", ".htm") if p.is_file()]

    ## SMY: TODO: convert htmls to PDF. Marker will by default attempt weasyprint which typically raise 'libgobject-2' error on Win
    
    return htmls

def collect_markdown_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all Markdown files.
    """
    root = Path(root)
    md_files = [p for p in root.rglob("*.md") if p.is_file()]
    return md_files

#m __future__ import annotations
def write_markdown(
    src_path: Union[str, Path],
    output_dir: Union[str, Path],
    rendered: Any,
) -> Path:
    
    """
    Write the Markdown representation of a source file to an output directory.

    Parameters
    ----------
    src_path : str | Path
        Path to the original source file. Only its base name is used for naming
        the resulting Markdown file.
    output_dir : str | Path
        Directory where the Markdown file will be written. It was created if it does not
        exist with create_outputdir().
    rendered : object
        Object that provides a ``markdown`` attribute containing the text to write.

    Returns
    -------
    pathlib.Path
        The full path of the written Markdown file.

    Raises
    ------
    FileNotFoundError
        If *src_path* does not point to an existing file.
    OSError
        If writing the file fails for any reason (e.g. permission denied).
    AttributeError
        If *rendered* does not expose a ``markdown`` attribute.

    Notes
    -----
    The function is intentionally lightweight: it only handles path resolution,
    directory creation, and file I/O. All rendering logic should be performed before
    calling this helper.
    """
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    #out_dir = Path(output_dir)
    #out_dir.mkdir(parents=True, exist_ok=True)

    md_name = f"{src.stem}.md"
    if isinstance(output_dir, Path):
        md_path = output_dir / f"{src.stem}" / md_name
    else:
        #md_path = Path(src.parent) / f"{Path(output_dir).stem}" / f"{src.stem}" / md_name
        
        ## Opt to putting output within same output_md folder rather than individual source folders
        #md_path = Path("data\\pdf") / "output_md" / f"{src.stem}" / md_name  ##debug
        md_path = Path("data") / output_dir / f"{src.stem}" / md_name  ##debug
    ##SMY: [resolved] Permission Errno13 - https://stackoverflow.com/a/57454275
    md_path.parent.mkdir(mode=0o2644, parents=True, exist_ok=True)  ##SMY: create nested md_path if not exists
    md_path.parent.chmod(0)

    try:
        markdown_text = getattr(rendered, "markdown")  ##SMY: get extracted markdown
    except AttributeError as exc:  # pragma: no cover
        raise AttributeError(
            "Extractor Rendered object must have a 'markdown' attribute"
        ) from exc

    with md_path.open(mode="w", encoding="utf-8") as md_f:
        md_f.write(markdown_text)    ##SMY: write markdown content to markdown file

    return md_path               ##SMY: return the markdown file  #✓ 
    #return {"files": md_path}   ##SMY: return dict of file with markdown filename.

# Dummp Markdown extracted images
def dump_images(
    src_path: Union[str, Path],
    output_dir: Union[str, Path],
    rendered: Any,
) -> int:
    
    """
    Dump the images  of the Markdown representation of a source file to an output directory.

    Parameters
    ----------
    src_path : str | Path
        Path to the original source file. Only its base name is used for naming
        the resulting Markdown file.
    output_dir : str | Path
        Directory where the Markdown file will be written. It was created if it does not
        exist with create_outputdir().
    rendered : object
        Object that provides a ``markdown`` attribute containing the text to write.

    Returns
    -------
    Number of images dumped from the  Markdown file.
    """

    try:
        images: Image.Image = getattr(rendered, "images")
    except TypeError as exc:  # pragma: no cover
        raise AttributeError(
            "Extracted images from rendered.images must be a mapping of str -> PIL.Image"
        ) from exc
    
    # Initialise variables
    images_count = 0
    img_path_list = []
    ##SMY: See marker.output.save_output()  : https://github.com/datalab-to/marker/blob/master/marker/output.py
    #for img_name, img_bytes in images.items():

    src = Path(src_path)  ##SMY: keep uniform with write_markdown. No need is exists anymore
    for img_name, img in images.items():
        # Resolve the full path and make sure any sub‑directories exist.
        #img_path = Path(output_dir) / src_path / img_name    ##SMY: image files  ##concatenate Path + str
        #img_path = create_outputdir(src_path) / img_name
        
        if isinstance(output_dir, Path):
            img_path = output_dir.stem / img_name            
        else:
            # #img_path = Path(output_dir) / f"{src.stem}" / img_name   ##SMY: create markdown file ##SMY concatenating Path with str
            # #img_path = Path(output_dir) / img_name   ##SMY: create markdown file ##SMY concatenating Path with str
            #img_path = Path(src.parent) / f"{Path(output_dir).stem}" / f"{src.stem}" / img_name
            
            #img_path = Path("data\\pdf") / "output_md" / f"{src.stem}" / img_name  ##debug
            img_path = Path("data") / output_dir / f"{src.stem}" / img_name  ##debug
        #img_path.mkdir(mode=0o777, parents=True, exist_ok=True)  ##SMY: create nested img_path if not exists
        #img_path.parent.mkdir(parents=True, exist_ok=True)

        img.save(img_path)    ##SMY: save images (of type PIL.Image.Image) to markdown folder
        images_count += 1
        #img_path_list = img_path_list.append(img_path)
        img_path_list.append(img_path)

    return images_count, img_path_list        ##SMY: return number of images and path
    #return images.items().count
    #return len(images)

# Dummp Markdown extracted images  ##SMY: Marked for deprecated
'''
def dump_images(
    src_path: Union[str, Path],
    output_dir: Union[str, Path],
    rendered: Any,
) -> int:
    
    """
    Dump the images  of the Markdown representation of a source file to an output directory.

    Parameters
    ----------
    src_path : str | Path
        Path to the original source file. Only its base name is used for naming
        the resulting Markdown file.
    output_dir : str | Path
        Directory where the Markdown file will be written. It was created if it does not
        exist with create_outputdir().
    rendered : object
        Object that provides a ``markdown`` attribute containing the text to write.

    Returns
    -------
    Number of images dumped from the  Markdown file.
    """

    try:
        images: Mapping[str, bytes] = getattr(rendered, "images")
    except TypeError as exc:  # pragma: no cover
        raise AttributeError(
            "Extracted images from rendered.images must be a mapping of str -> bytes"
        ) from exc

    images_count = 0
    ##SMY: See marker.output.save_output()  : https://github.com/datalab-to/marker/blob/master/marker/output.py
    #for img_name, img_bytes in images.items():
    for img_name, img in images.items():
        # Resolve the full path and make sure any sub‑directories exist.
        img_path = Path(output_dir) / src_path / img_name    ##SMY: image files  ##concatenate Path + str
        img_path.parent.mkdir(parents=True, exist_ok=True)

        #'' '
        #with img_path.open("wb") as fp:
        #    fp.write(img_bytes)    ##SMY: write images to markdown folder
        #images_count += 1
        #'' '
        img.save(img_path)    ##SMY: save images (of type PIL.Image.Image) to markdown folder
        images_count += 1

    return images_count        ##SMY: return number of images
    #return images.items().count
    #return len(images)
'''