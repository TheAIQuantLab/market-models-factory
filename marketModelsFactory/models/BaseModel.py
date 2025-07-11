from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import marketModelsFactory.utils.plotting_utils as pltu

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

    def plot_paths(self, data=None, n_paths: int = 10):
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
        pltu.plot_paths(data, title="Simulated GBM Paths", n_paths=n_paths)

    def plot_histogram(self, data=None, nbins: int = 20, fit_normal: bool = False, data_fit=None):
        """
        Plot histogram of terminal percentage returns from simulated data,
        and optionally overlay fitted historical returns and normal distribution.

        Parameters
        ----------
        data : np.ndarray, optional
            Simulated paths of shape (n_paths, n_steps). If None, it will be generated.
        nbins : int, optional
            Number of bins in the histogram. Default is 20.
        fit_normal : bool, optional
            Whether to overlay a fitted normal distribution on top of the histogram.
        data_fit : np.ndarray, optional
            Real historical price series to compare terminal percentage returns with.

        Returns
        -------
        None
            Displays a histogram of terminal percentage returns and optionally comparisons.
        """
        if data is None:
            data = self.generate_synthetic_data(n_paths=1000, n_steps=252)

        n_steps = data.shape[1] - 1
        plt.figure(figsize=(8, 5))

        if data_fit is not None:
            # Compare real percentage returns over same time interval
            if len(data_fit) > n_steps:
                pct_returns = (data_fit[n_steps:] - data_fit[:-n_steps]) / data_fit[:-n_steps]
                plt.hist(pct_returns, bins=nbins, histtype='step', alpha=0.7,
                         color='black', label='Fitted Data % Returns', density=True)

        if fit_normal:
            pltu.plot_fitted_normal(data)

        pltu.plot_terminal_histogram(data, nbins=nbins)
        plt.title("Histogram of Terminal Values (Stochastic Model)")
        plt.xlabel("Terminal Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
