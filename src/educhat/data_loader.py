import pathlib
from typing import List

def load_txt_files(folder: str) -> List[str]:
    base = pathlib.Path(folder)
    texts = []
    for path in base.rglob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return texts
