from dataclasses import dataclass

@dataclass
class DataGenerationParams:
    n_samples: int = 100
    slope: float = 2.0
    intercept: float = 1.0
    noise_std: float = 1.0
    random_seed: int = 42
    training_fraction: float = 0.8

    def __post_init__(self):
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")

        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")

        if not (0 < self.training_fraction < 1):
            raise ValueError(f"training_fraction must be between 0 and 1, got {self.training_fraction}")
