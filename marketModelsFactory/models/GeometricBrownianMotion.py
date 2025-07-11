from marketModelsFactory.models.BaseModel import BaseModel
import numpy as np

class GeometricBrownianMotion(BaseModel):
    def __init__(self, params: dict):
        super().__init__(params)
        self.mu = params.get("mu", 0.05)       # Annualized drift
        self.sigma = params.get("sigma", 0.2)  # Annualized volatility
        self.S0 = params.get("S0", 100)
        self.dt = params.get("dt", 1/252)      # Daily step (default)

    def fit(self, data):
        pct_returns = np.diff(data) / data[:-1]
        mu_daily = np.mean(pct_returns)
        sigma_daily = np.std(pct_returns, ddof=1)

        # Annualize
        self.mu = mu_daily / self.dt
        self.sigma = sigma_daily / np.sqrt(self.dt)

    def generate_synthetic_data(self, n_paths: int, n_steps: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros((n_paths, n_steps))
        S[:, 0] = self.S0

        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)

        for t in range(1, n_steps):
            z = np.random.randn(n_paths)
            S[:, t] = S[:, t - 1] * (1 + np.exp(drift + diffusion * z) - 1)

        return S
