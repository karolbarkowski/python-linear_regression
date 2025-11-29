"""
Steps package for Linear Regression Tutorial
Contains modular step functions for the linear regression workflow
"""

from .generate_linear_data import generate_linear_data
from .split_data import split_data
from .train_sklearn import train_sklearn
from .train_manual import train_manual
from .eval import evaluate_model
from .visualize import visualize

__all__ = [
    'generate_linear_data',
    'split_data',
    'train_sklearn',
    'train_manual',
    'evaluate_model',
    'visualize'
]
