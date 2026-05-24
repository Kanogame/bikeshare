from pathlib import Path

from src.low.creds.reader import CredsReader


def load_creds(path: str) -> CredsReader:
    return CredsReader.load(Path(path))
