"""IO module for reading Fast5/POD5 sequencing data."""
from .fast5_reader import Fast5Reader, StreamingFast5Reader

__all__ = ["Fast5Reader", "StreamingFast5Reader"]
