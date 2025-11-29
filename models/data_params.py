
class DataGenerationParams:
   
    def __init__(
        self,
        n_samples: int = 100,
        slope: float = 2.0,
        intercept: float = 1.0,
        noise_std: float = 1.0,
        random_seed: int = 42,
        training_fraction: float = 0.8
    ):
      
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        
        if not (0 < training_fraction < 1):
            raise ValueError(f"training_fraction must be between 0 and 1, got {training_fraction}")

        self.n_samples = n_samples
        self.slope = slope
        self.intercept = intercept
        self.noise_std = noise_std
        self.random_seed = random_seed
        self.training_fraction = training_fraction
