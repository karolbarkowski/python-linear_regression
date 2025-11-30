"""
Mode enum for selecting regression training method
"""
from enum import Enum


class Mode(Enum):
    """Training mode selection"""
    SKLEARN = "sklearn"
    MANUAL = "manual"
