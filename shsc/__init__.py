"""SHSC: spatial hypoxia score combined with clinical variables."""

from .config import Paths
from .pipeline import run

__all__ = ["Paths", "run"]
__version__ = "1.0.0"
