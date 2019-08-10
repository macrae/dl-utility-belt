"General Utility Methods"

import glob


def append_files_from_path(path: str) -> str:
    """Given a path to a list of .txt files return a string
    of the concatenated contents of the file."""
    paths = glob.glob(path + "/*.txt")
    text = ""
    for path in paths:
        with open(path, "r") as f:
            read_text = f.read()
            text += read_text
    return text
