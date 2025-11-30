"""
Training parameters for manual gradient descent
"""
from dataclasses import dataclass


@dataclass
class TrainingParams:
    """Parameters for manual linear regression training"""
    n_iterations: int = 1000
    learning_rate: float = 0.01
    loss_threshold: float = 1e-6

    def __post_init__(self):
        if self.n_iterations <= 0:
            raise ValueError(f"n_iterations must be positive, got {self.n_iterations}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.loss_threshold < 0:
            raise ValueError(f"loss_threshold must be non-negative, got {self.loss_threshold}")
