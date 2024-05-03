import os

from sbx.sac import SAC, VLM_SAC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__all__ = [
    "SAC",
    "VLM_SAC"
]
