"""This file compiles useful functions related to data and file handling."""

import os
import re


def get_files_path_recursively(path, *args, suffix=''):
    """Retrieve specific files path recursively from a directory.

    Retrieve the path of all files with one of the given extension names,
    in the given directory and all its subdirectories, recursively.
    The extension names should be given as a list of strings. The search for
    extension names is case sensitive.

    Args:
        path (str): root directory from which to search for files recursively
        *args: list of file extensions to be considered.

    Returns:
        list(str): list of paths of every file in the directory and all its
                   subdirectories.
    """
    exts = list(args)
    for ext_i, ext in enumerate(exts):
        exts[ext_i] = ext[1:] if ext[0] == '.' else ext
    ext_list = "|".join(exts)
    result = [os.path.join(dp, f)
              for dp, dn, filenames in os.walk(path)
              for f in filenames
              if re.search(rf"^.*({suffix})\.({ext_list})$", f)]
    return result
