from typing import Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    from utils.plotting import ResultsPlotter
except ImportError as e:
    raise ImportError(
        "Failed to import ResultsPlotter from utils.plotting. "
        "Ensure the project root is on PYTHONPATH."
    ) from e


ArrayLike = Union[np.ndarray, list, tuple]


class HDMFResultsPlotter(ResultsPlotter):
    """
    Plotting utilities for HDMF simulation outputs.

    Inherits global styling, configuration, and helpers from ResultsPlotter,
    and adds:
      - plot_fc:  plot a functional connectivity matrix
      - plot_fcd: plot a functional connectivity dynamics matrix
      - plot_rates: plot raw firing rates (supports time_fraction)
      - plot_bold:  plot raw BOLD (supports time_fraction)
    """
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        # Color palette for FC plots
        self.fc_cmap = plt.get_cmap("bwr")
        self.fcd_cmap = plt.get_cmap("mako")
        self.colors.update({
            "rate": "#592E83",  # purple
            "bold": "#C73E1D",  # orange-red
        })
        self.TITLE = 14
        self.LABEL = 12
        self.TICKS = 10
        self.fc_color_palette = plt.get_cmap("bwr")
    # ---------------------- Matrix plots (FC / FCD) ---------------------- #
    def plot_fc(
        self,
        fc: ArrayLike,
        title: Optional[str] = None,
        cmap: str = "bwr",
        zscore_offdiag: bool = False,        
        colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot a Functional Connectivity (FC) matrix.

        Parameters:
            fc: 2D array (N x N)
            zscore_offdiag: if True, z-score using off-diagonal values only
            vmin, vmax: color limits; if None, use symmetric max(abs()) around 0
        """
        mat = np.asarray(fc)
        if mat.ndim != 2:
            raise ValueError(f"FC must be 2D (N x N), got shape {mat.shape}")

        plot_mat = mat.copy()
        if zscore_offdiag:
            iu = np.triu_indices_from(plot_mat, k=1)
            vals = plot_mat[iu]
            if np.std(vals) > 0:
                z = (plot_mat - np.mean(vals)) / np.std(vals)
                # keep diagonal unscaled
                np.fill_diagonal(z, np.diag(plot_mat))
                plot_mat = z
            else:
                warnings.warn("Off-diagonal std is zero; skipping z-scoring.")

        if vmin is None or vmax is None:
            lim = np.max(np.abs(plot_mat))
            vmin = -lim
            vmax = lim

        fig, _ = plt.subplots(figsize=(7.55, 6))

        h = plt.imshow(plot_mat, interpolation='none', aspect='auto', cmap=self.fc_color_palette, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(h)
        # Put -1 and 1 in the colorbar
        cbar.set_ticks([-lim, lim])
        cbar.set_ticklabels(['-5', '5'], fontsize=self.TITLE)
        cbar.set_label('Correlation (z-score)', fontsize=self.TITLE)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        return fig

    def plot_fcd(
        self,
        fcd: ArrayLike,
        title: Optional[str] = None,
        cmap: str = "mako",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot a Functional Connectivity Dynamics (FCD) matrix.

        Parameters:
            fcd: 2D array (nWins x nWins)
        """
        mat = np.asarray(fcd)
        if mat.ndim != 2:
            raise ValueError(f"FCD must be 2D (nWins x nWins), got shape {mat.shape}")

        if vmin is None or vmax is None:
            vmin, vmax = np.min(mat), np.max(mat)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", aspect="auto")
        ax.set_title(title or "Functional Connectivity Dynamics")
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        

    # ---------------------- Time series plots (Rates / BOLD) ---------------------- #
    def plot_rates(
        self,
        rates: ArrayLike,
        time_fraction: float = 1.0,
        mode: str = "mean",            # "mean" or "region"
        region_index: int = 0,         # used if mode == "region"
        downsample: int = 1,
        title: Optional[str] = None,
        xlabel: str = "Time (a.u.)",
        ylabel: str = "Firing rate (Hz)",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot raw firing rates over time.

        Parameters:
            rates: 1D (T,), 2D, or 3D array. Heuristics:
                   - If 2D, assumes last axis is time; averages across other axes in mode='mean'
                   - If 3D (e.g., reps x regions x T), averages across non-time axes in mode='mean'
            time_fraction: fraction of the full time series to display (0 < f <= 1)
            mode: 'mean' to average across non-time axes, or 'region' to plot a single region
            region_index: which region to plot if mode='region'
            downsample: keep every Nth sample for plotting speed
        """
        sig = self._extract_1d_timeseries(rates, time_fraction, mode, region_index, downsample)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(sig.shape[0]), sig, color=self.colors.get("rate", "#592E83"), linewidth=1.5)
        ax.set_xlim(0, sig.shape[0] - 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "Firing rates (raw)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        return fig

    def plot_bold(
        self,
        bold: ArrayLike,
        time_fraction: float = 1.0,
        mode: str = "mean",            # "mean" or "region"
        region_index: int = 0,         # used if mode == "region"
        downsample: int = 1,
        title: Optional[str] = None,
        xlabel: str = "Time (a.u.)",
        ylabel: str = "BOLD (a.u.)",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot raw BOLD over time.

        Parameters:
            bold: 1D (T,), 2D, or 3D array; same handling as plot_rates
            time_fraction: fraction of the full time series to display (0 < f <= 1)
            mode: 'mean' to average across non-time axes, or 'region' to plot a single region
            region_index: which region to plot if mode='region'
            downsample: keep every Nth sample for plotting speed
        """
        sig = self._extract_1d_timeseries(bold, time_fraction, mode, region_index, downsample)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(sig.shape[0]), sig, color=self.colors.get("bold", "#C73E1D"), linewidth=1.5)
        ax.set_xlim(0, sig.shape[0] - 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "BOLD (raw)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        return fig
    # plot for bold and rates at the same time
    def plot_bold_and_rates(self,
        bold: ArrayLike,
        rates: ArrayLike,
        time_fraction: float = 1.0,
        mode: str = "mean",            # "mean" or "region"
        region_index: int = 0,         # used if mode == "region"
        downsample: int = 1,
        title: Optional[str] = None,
        xlabel: str = "Time (a.u.)",
        ylabel_bold: str = "BOLD (a.u.)",
        ylabel_rates: str = "Firing rate (Hz)",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot raw BOLD and firing rates over time.

        Parameters:
            bold: 1D (T,), 2D, or 3D array; same handling as plot_rates
            rates: 1D (T,), 2D, or 3D array; same handling as plot_rates
            time_fraction: fraction of the full time series to display (0 < f <= 1)
            mode: 'mean' to average across non-time axes, or 'region' to plot a single region
            region_index: which region to plot if mode='region'
            downsample: keep every Nth sample for plotting speed
        """
        

        fig, ax1 = plt.subplots(figsize=(10, 4))

        color_bold = self.colors.get("bold", "#C73E1D")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_rates = self.colors.get("rate", "#592E83")
        ax2.set_ylabel(ylabel_rates, color=color_rates)
        ax2.plot(np.arange(rates.shape[0]), rates, color=color_rates, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color_rates)
        ax2.grid(False)

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel_bold, color=color_bold)
        ax1.plot(np.arange(0,rates.shape[0],self.config['simulation']['TR']*1/0.001), bold, color=color_bold, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color_bold)
        ax1.grid(True, alpha=0.3)        
        ax1.set_title(title or "BOLD and Firing rates (raw)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            return fig

    # ---------------------- Helpers ---------------------- #
    def _extract_1d_timeseries(
        self,
        data: ArrayLike,
        time_fraction: float,
        mode: str,
        region_index: int,
        downsample: int,
    ) -> np.ndarray:
        """
        Convert input (1D/2D/3D) to a single 1D time series, slicing by time_fraction.
        Heuristic: the longest axis is treated as time.
        """
        arr = np.asarray(data)
        if arr.ndim == 0:
            raise ValueError("Timeseries data is scalar; expected array.")
        if arr.ndim == 1:
            ts = arr
        else:
            # Move time axis to last: assume longest axis is time
            time_axis = int(np.argmax(arr.shape))
            arr = np.moveaxis(arr, time_axis, -1)  # (..., T)
            T = arr.shape[-1]
            if not (0 < time_fraction <= 1.0):
                raise ValueError("time_fraction must be in (0, 1].")
            t_end = max(1, int(np.round(T * time_fraction)))
            arr = arr[..., :t_end]

            if mode not in ("mean", "region"):
                raise ValueError("mode must be 'mean' or 'region'.")

            if arr.ndim == 2:
                # (signals, T)
                if mode == "mean":
                    ts = arr.mean(axis=0)
                else:
                    if not (0 <= region_index < arr.shape[0]):
                        raise IndexError(f"region_index {region_index} out of range [0, {arr.shape[0]-1}]")
                    ts = arr[region_index]
            elif arr.ndim == 3:
                # (something, signals, T) e.g., (reps, regions, T)
                if mode == "mean":
                    ts = arr.mean(axis=(0, 1))
                else:
                    # pick region across first dim average
                    if not (0 <= region_index < arr.shape[1]):
                        raise IndexError(f"region_index {region_index} out of range [0, {arr.shape[1]-1}]")
                    ts = arr[:, region_index, :].mean(axis=0)
            else:
                # Higher dims: average across all non-time axes in 'mean' mode
                if mode == "mean":
                    axes = tuple(range(arr.ndim - 1))
                    ts = arr.mean(axis=axes)
                else:
                    warnings.warn(
                        "mode='region' not supported for >3D input; falling back to mean across non-time axes."
                    )
                    axes = tuple(range(arr.ndim - 1))
                    ts = arr.mean(axis=axes)

        if downsample and downsample > 1:
            ts = ts[::downsample]
        return ts.astype(np.float32, copy=False)