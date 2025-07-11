import numpy as np
import matplotlib.pyplot as plt

def plot_paths(data: np.ndarray, title: str = "Simulated Paths", n_paths: int = 10, color: str = 'blue') -> None:
    """
    Plot multiple simulated paths from a time series dataset.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (n_paths, n_steps), where each row is a simulated path.
    title : str, optional
        Title of the plot. Default is "Simulated Paths".
    n_paths : int, optional
        Number of paths to plot. Default is 10.
    color : str, optional
        Color of the plotted paths. Default is 'blue'.

    Returns
    -------
    None
        Displays a matplotlib line plot of the simulated paths.
    """
    plt.figure(figsize=(10, 5))
    for i in range(min(n_paths, data.shape[0])):
        plt.plot(data[i], color=color, alpha=0.5)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_terminal_histogram(data: np.ndarray, nbins: int = 20, label: str = 'Terminal % Returns', color: str = 'blue') -> None:
    """
    Plot a histogram of terminal percentage returns across simulated paths.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (n_paths, n_steps), where each row is a simulated path.
    nbins : int, optional
        Number of bins in the histogram. Default is 20.
    label : str, optional
        Label for the histogram. Default is 'Terminal % Returns'.
    color : str, optional
        Color of the histogram bars. Default is 'blue'.

    Returns
    -------
    None
        Displays a matplotlib histogram of terminal percentage returns.
    """
    terminal_pct_returns = (data[:, -1] - data[:, 0]) / data[:, 0]
    plt.hist(terminal_pct_returns, bins=nbins, alpha=0.7, edgecolor='black', 
             density=True, label=label, color=color)


def plot_fitted_normal(data: np.ndarray) -> None:
    """
    Fit and overlay a normal distribution on terminal percentage returns.

    Parameters
    ----------
    data : np.ndarray
        A 2D NumPy array of shape (n_paths, n_steps), representing simulated asset paths.

    Returns
    -------
    None
        Displays a line plot of the fitted normal distribution over existing histogram.
    """
    terminal_pct_returns = (data[:, -1] - data[:, 0]) / data[:, 0]
    mu = np.mean(terminal_pct_returns)
    std = np.std(terminal_pct_returns)

    x = np.linspace(mu - 4 * std, mu + 4 * std, 1000)
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    
    plt.plot(x, pdf, color='red', label='Fitted Normal Distribution', linewidth=2)
