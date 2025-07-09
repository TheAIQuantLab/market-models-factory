from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def fit(self, data):
        """Estimate model parameters from data."""
        pass

    @abstractmethod
    def generate_synthetic_data(self, n_paths: int, n_steps: int, **kwargs):
        """Simulate synthetic paths."""
        pass

    @abstractmethod
    def plot_paths(self, data=None, **kwargs):
        """Plot the simulated paths."""
        pass

    @abstractmethod
    def plot_histogram(self, data=None, **kwargs):
        """Plot histogram of terminal values or increments."""
        pass
