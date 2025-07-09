from marketModelsFactory.models.model_base import BaseModel
import numpy as np
import matplotlib.pyplot as plt

class BrownianMotionModel(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.mu = params.get("mu", 0.05)
        self.sigma = params.get("sigma", 0.2)
        self.S0 = params.get("S0", 100)

    def fit(self, data):
        """
        Fit μ and σ using log returns of price data.
        """
        log_returns = np.diff(np.log(data))
        self.mu = np.mean(log_returns)
        self.sigma = np.std(log_returns)

    def generate_synthetic_data(self, n_paths: int, n_steps: int, dt: float = 1/252, seed: int = None):
        """
        Simulate GBM paths using Euler-Maruyama.
        """
        if seed is not None:
            np.random.seed(seed)
        
        S = np.zeros((n_paths, n_steps))
        S[:, 0] = self.S0
        
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps):
            z = np.random.randn(n_paths)
            S[:, t] = S[:, t - 1] * np.exp(drift + diffusion * z)

        return S

    def plot_paths(self, data=None, n_paths: int = 10):
        """
        Plot simulated GBM paths.
        """
        if data is None:
            data = self.generate_synthetic_data(n_paths=n_paths, n_steps=252)
        plt.figure(figsize=(10, 5))
        for i in range(min(n_paths, data.shape[0])):
            plt.plot(data[i])
        plt.title("Simulated GBM Paths (Black-Scholes)")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, data=None):
        """
        Plot histogram of terminal prices.
        """
        if data is None:
            data = self.generate_synthetic_data(n_paths=1000, n_steps=252)
        terminal_prices = data[:, -1]
        plt.figure(figsize=(8, 5))
        plt.hist(terminal_prices, bins=50, alpha=0.7, edgecolor='black')
        plt.title("Histogram of Terminal Prices (GBM)")
        plt.xlabel("Terminal Price")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
