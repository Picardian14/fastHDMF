from typing import Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from .experiment_manager import ExperimentManager
try:
    from .utils.plotting import ResultsPlotter
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
    def __init__(
        self,
        experiment_manager: ExperimentManager,        
        style_config: Optional[dict] = None,
        style_config_path: Optional[Union[str, Path]] = None,
    ):
        # Initialize with plotting style and experiment configurations
        self.experiment_manager = experiment_manager
        super().__init__(            
            config=experiment_manager.current_config,
            config_path=experiment_manager.current_config_path,
            style_config=style_config,
            style_config_path=style_config_path,
        )
        # Color palette for FC plots
        self.fc_cmap = plt.get_cmap("bwr")
        self.fcd_cmap = plt.get_cmap("mako")
        self.colors.update({
            "rate": "#592E83",  # purple
            "bold": "#C73E1D",  # orange-red
        })
        self.TITLE = self.style_config['matplotlib']['rcParams'].get('axes.titlesize', 24)
        self.LABEL = self.style_config['matplotlib']['rcParams'].get('axes.labelsize', 20)
        self.TICKS = self.style_config['matplotlib']['rcParams'].get('xtick.labelsize', 20)
        self.fc_color_palette = plt.get_cmap("bwr")
        # Defaults for figure text sizes (can be overridden by config or caller)
        self.COLORBAR = self.style_config['matplotlib']['rcParams'].get('colorbar.size', 20)
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

    # ---------------------- Homeostatic fit plot ---------------------- #
    def plot_homeostatic_fit(
        self,
        hom_fit_results,
        obj_rates: Optional[list] = None,
        lr_param: str = "lrj",
        decay_param: str = "taoj",
        g_param: str = "G",
        target_index: int = 1,
        cmap: str = "seismic",
        vmin: float = -100,
        vmax: float = 100,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Plot the average homeostatic fit over G for a selected objective rate.

        Inputs contract:
                - hom_fit_results: list or dict of 3D arrays of mismatch values across (G, LR, DECAY) axes.
                    Axis order can vary; the method will infer axes by matching lengths to parameter ranges.
                    For multiple objective rates, provide a list (ordered) or dict keyed by rate value.
        - obj_rates: list of target firing rates corresponding to the entries. If None and
          hom_fit_results is a dict, keys will be used (sorted ascending).

        Behavior:
        - Computes mean and std over G, finds the LR index minimizing |mean mismatch| across DECAY,
          fits a linear trend of min positions vs LR index, and plots heatmap of % mismatch.
        - Axis ticks/labels are inferred from config grid ranges for lr_param and decay_param when available.
        """
        # Normalize inputs to a list of arrays and a list of objective rates
        if isinstance(hom_fit_results, dict):
            # sort by key if keys are numeric (rates)
            try:
                keys_sorted = sorted(hom_fit_results.keys(), key=lambda x: float(x))
            except Exception:
                keys_sorted = list(hom_fit_results.keys())
            arr_list = [np.asarray(hom_fit_results[k]) for k in keys_sorted]
            obj_list = [float(k) for k in keys_sorted]
        elif isinstance(hom_fit_results, (list, tuple)):
            arr_list = [np.asarray(a) for a in hom_fit_results]
            obj_list = obj_rates if obj_rates is not None else list(range(len(arr_list)))
        else:
            arr_list = [np.asarray(hom_fit_results)]
            obj_list = obj_rates if obj_rates is not None else [self.config.get('simulation', {}).get('obj_rate', 3.44)]

        # Parameter ranges from config (fallbacks if missing)
        def _range_from_config(name: str, default: Optional[np.ndarray] = None) -> np.ndarray:
            # Prefer ranges computed by parent class (utils.plotting extracts grid ranges)
            if hasattr(self, 'parameter_ranges') and name in getattr(self, 'parameter_ranges', {}):
                return np.asarray(self.parameter_ranges[name])
            # Check explicit Homeostatic_Grid-like config where functions are used
            grid_cfg = self.config.get('grid', {})
            if name in grid_cfg:
                cfg = grid_cfg[name]
                # If using function def: {fun: 'np.logspace', args: [...]}
                if isinstance(cfg, dict) and 'fun' in cfg:
                    fun = cfg['fun']
                    args = cfg.get('args', [])
                    if fun == 'np.logspace':
                        return np.logspace(*args)
                    elif fun == 'np.linspace':
                        return np.linspace(*args)
                # If using start/end/step
                if isinstance(cfg, dict) and all(k in cfg for k in ('start', 'end', 'step')):
                    start, end, step = cfg['start'], cfg['end'], cfg['step']
                    # inclusive-ish end
                    return np.arange(start, end + step/2, step)
                # If list of values
                if isinstance(cfg, (list, tuple)):
                    return np.asarray(cfg)
            # Default
            if default is not None:
                return np.asarray(default)
            # Reasonable fallbacks matching the notebook
            if name == 'G':
                return np.arange(0, 8.5, 0.5)
            if name == lr_param:
                return np.logspace(0, 2, 100)
            if name == decay_param:
                return np.logspace(2, 5, 110)
            return np.arange(0, 1, 1)

        G_range = _range_from_config(g_param)
        LR_range = _range_from_config(lr_param)
        DECAY_range = _range_from_config(decay_param)

        # Determine pure-decades ticks (first three decades) for LR and DECAY
        exp_min_lr = int(np.floor(np.log10(LR_range[0])))
        lr_ticks = []
        lr_labels = []
        for e in range(exp_min_lr, exp_min_lr + 3):
            val = 10**e
            idx = int(np.argmin(np.abs(LR_range - val)))
            lr_ticks.append(idx)
            lr_labels.append(f"{val:g}")

        exp_min_dec = int(np.floor(np.log10(DECAY_range[0])))
        decay_ticks = []
        decay_labels = []
        for e in range(exp_min_dec, exp_min_dec + 3):
            val = 10**e
            idx = int(np.argmin(np.abs(DECAY_range - val)))
            decay_ticks.append(idx)
            decay_labels.append(f"{val:g}")

        # Process arrays: detect G/LR/DECAY axes, average over G, ensure orientation (DECAY, LR)
        mean_hom_fit = []
        std_hom_fit = []
        nlr = len(LR_range)
        ndec = len(DECAY_range)
        nG = len(G_range)
        for a in arr_list:
            if a.ndim != 3:
                raise ValueError(f"Each homeostatic grid must be 3D, got {a.shape}")
            # find g axis by size match
            axes_sizes = a.shape
            try:
                g_axis = [i for i, s in enumerate(axes_sizes) if s == nG]
                g_axis = g_axis[0]
            except Exception:
                # fallback: assume last axis is G (as in notebook)
                g_axis = 2
            a_mean = np.mean(a, axis=g_axis)
            # Now a_mean is 2D, remaining axes correspond to LR and DECAY (order unknown)
            if a_mean.shape == (ndec, nlr):
                arr2d = a_mean
            elif a_mean.shape == (nlr, ndec):
                arr2d = a_mean.T
            else:
                # attempt to identify by nearest sizes (robust against off-by-one)
                if abs(a_mean.shape[0] - ndec) + abs(a_mean.shape[1] - nlr) <= \
                   abs(a_mean.shape[0] - nlr) + abs(a_mean.shape[1] - ndec):
                    arr2d = a_mean
                else:
                    arr2d = a_mean.T
            mean_hom_fit.append(arr2d)
            std_hom_fit.append(np.std(a, axis=g_axis).T if arr2d.shape != np.std(a, axis=g_axis).shape else np.std(a, axis=g_axis))

        # Extract minimum mismatch positions along DECAY for each LR index
        # In notebook they used argmin(abs(x)) over axis=0, where x is mean over G -> (LR, DECAY)
        min_mm_pos = [np.argmin(np.abs(x), axis=0) for x in mean_hom_fit]  # shape (LR,)

        # Fit linear trend of min positions vs LR index (replicate logic with special-case for first target)
        fit_res = []
        fit_res_2plot = []
        for o in range(len(arr_list)):
            x_idx = np.arange(nlr)
            y = min_mm_pos[o]
            if o == 0 and nlr > 21:  # mimic notebook's slice
                coeff = np.polyfit(LR_range[21:], y[21:], 1)
                coeff_plot = np.polyfit(x_idx[21:], y[21:], 1)
            else:
                coeff = np.polyfit(LR_range, y, 1)
                coeff_plot = np.polyfit(x_idx, y, 1)
            fit_res.append(coeff)
            fit_res_2plot.append(coeff_plot)

        # Choose which objective rate to plot
        o = int(np.clip(target_index, 0, len(arr_list) - 1))
        obj_rate = obj_list[o]

    # Create plot
        fig = plt.figure()
        # percent mismatch relative to target
        heat = 100.0 * mean_hom_fit[o] / float(obj_rate)
        im = plt.imshow(
            heat,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin='lower',
            interpolation='none',
            aspect='auto',
        )

        # Overlay min points and fitted line in LR-index coordinates
        x_idx = np.arange(nlr)
        plt.plot(x_idx, min_mm_pos[o], '.', label='Min', color='limegreen')
        plt.plot(
            x_idx,
            fit_res_2plot[o][0] * x_idx + fit_res_2plot[o][1],
            '-', color='darkgreen',
            label=f"y={fit_res[o][0]:.3f} x + {fit_res[o][1]:.3f}"
        )

        # X ticks: fixed LR indices for 1,10,100
        xticks = [0, nlr//2, nlr-1]
        xlabels = ['1', '10', '100']
        plt.xticks(ticks=xticks, labels=xlabels, fontsize=self.TICKS)
        plt.xlabel('Learning Rate', fontsize=self.LABEL)
        plt.title(f"Mean homeostatic fit for target: {obj_rate} Hz", fontsize=self.TITLE)
        plt.ylim([0, ndec - 1])
        plt.legend(fontsize=self.TICKS)

        # Y ticks: fixed DECAY indices for 100,1000,10000
        yticks = [0, ndec//2, ndec-1]
        ylabels = ['100', '1000', '10000']
        plt.yticks(ticks=yticks, labels=ylabels, fontsize=self.TICKS)
        plt.ylabel('Decay', fontsize=self.LABEL)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('% of Firing Rate Mismatch', fontsize=self.COLORBAR)
        cbar.ax.tick_params(labelsize=self.TICKS)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
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