"""
Steps package for Linear Regression Tutorial
Contains modular step functions for the linear regression workflow
"""

from .generate_linear_data import generate_linear_data
from .split_data import split_data
from .train import train
from .eval import evaluate_model
from .visualize import visualize

__all__ = [
    'generate_linear_data',
    'split_data',
    'train',
    'evaluate_model',
    'visualize'
]
