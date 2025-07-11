from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel(ABC):
    """
    Abstract base class for financial time series models.

    Provides a consistent interface for fitting to data,
    simulating synthetic paths, and visualizing results.
    """

    def __init__(self, params: dict):
        """
        Initialize the model with a dictionary of parameters.

        Parameters:
        ----------
        params : dict
            Dictionary of model parameters.
        """
        self.params = params

    @abstractmethod
    def fit(self, data):
        """
        Estimate model parameters from historical data.

        Parameters:
        ----------
        data : array-like
            Time series data (e.g., historical prices) to calibrate the model.
        """
        pass

    @abstractmethod
    def generate_synthetic_data(self, n_paths: int, n_steps: int, **kwargs):
        """
        Simulate synthetic paths using the model.

        Parameters
        -------
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps per path.
        **kwargs : dict
            Additional arguments for model-specific simulation.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_paths, n_steps) containing simulated values.
        """
        pass

    def plot_paths(self, data=None, n_paths: int = 10, title: str = "Simulated Paths"):
        """
        Plot a selection of simulated paths.

        Parameters
        ----------
        data : np.ndarray, optional
            Precomputed synthetic data. If None, it will be generated.
        n_paths : int, optional
            Number of paths to plot. Default is 10.

        Returns
        -------
        None
            Displays a line plot of the simulated paths.
        """
        if data is None:
            data = self.generate_synthetic_data(n_paths=n_paths, n_steps=252)
        plt.figure(figsize=(10, 5))
        for i in range(min(n_paths, data.shape[0])):
            plt.plot(data[i], color='blue', alpha=0.5)
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, data=None, nbins: int = 20, fit_normal: bool = False, data_fit=None):
        """
        Plot KDE of terminal percentage returns from simulated data,
        and optionally overlay fitted historical returns and normal distribution.

        Parameters
        ----------
        data : np.ndarray, optional
            Simulated paths of shape (n_paths, n_steps). If None, it will be generated.
        nbins : int, optional
            (Unused here unless histogram is preferred) - kept for compatibility.
        fit_normal : bool, optional
            Whether to overlay a fitted normal distribution.
        data_fit : np.ndarray, optional
            Real historical price series to compare returns with.

        Returns
        -------
        None
        """
        if data is None:
            data = self.generate_synthetic_data(n_paths=1000, n_steps=252)

        n_steps = data.shape[1] - 1
        terminal_pct_returns = (data[:, -1] - data[:, 0]) / data[:, 0]

        plt.figure(figsize=(8, 5))

        if data_fit is not None and len(data_fit) > n_steps:
            pct_returns = (data_fit[n_steps:] - data_fit[:-n_steps]) / data_fit[:-n_steps]
            sns.kdeplot(pct_returns, label='Fitted Data % Returns', color='black', linewidth=2, fill=True, alpha=0.2)

        sns.kdeplot(terminal_pct_returns, label='Simulated Terminal Returns', color='blue', linewidth=2, fill=True, alpha=0.2)

        if fit_normal:
            mu = np.mean(terminal_pct_returns)
            std = np.std(terminal_pct_returns)
            x = np.linspace(mu - 4 * std, mu + 4 * std, 1000)
            normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
            plt.plot(x, normal_pdf, color='red', linestyle='--', label='Fitted Normal PDF')

        plt.title("KDE of Terminal % Returns (Simulated vs Real)")
        plt.xlabel("Terminal Return")
        plt.ylabel("Density")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

