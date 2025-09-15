"""
Plotting library for VirtualFingerprint analysis.

This module provides a centralized plotting interface with consistent styling
and specialized plots for analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings

class ResultsPlotter:
    """
    Centralized plotter for main analysis with configuration-aware styling.
    
    This class stores experiment configuration information and provides
    consistent styling and labeling for various analysis plots.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the plotter with configuration and styling.
        
        Parameters:
            config (Dict[str, Any], optional): Configuration dictionary
            config_path (str, optional): Path to configuration YAML file
        """
        self.config = self._load_config(config, config_path)
        self._setup_style()
        self._extract_parameter_ranges()
        
    def _load_config(self, config: Optional[Dict[str, Any]], config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from dict or file."""
        if config is not None:
            return config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return empty config if none provided
            return {}
    
    def _setup_style(self):
        """Set up consistent styling for all plots."""
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Set matplotlib parameters for publication-quality plots
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Define color palette for different analysis types
        self.colors = {
            'identifiability': '#2E86AB',
            'structural_diff': '#A23B72',
            'correlation': '#F18F01',
            'bold': '#C73E1D',
            'rate': '#592E83',
            'gradient': '#1B4D3E'
        }
    
    def _extract_parameter_ranges(self):
        """Extract parameter ranges from configuration for labeling."""
        self.parameter_ranges = {}
        
        if 'grid' in self.config:
            grid_config = self.config['grid']
            
            # Extract G parameter range if it exists
            if 'G' in grid_config:
                g_config = grid_config['G']
                if isinstance(g_config, dict) and all(key in g_config for key in ['start', 'end', 'step']):
                    # Generate G values from start, end, step
                    start, end, step = g_config['start'], g_config['end'], g_config['step']
                    self.parameter_ranges['G'] = np.arange(start, end + step, step)  # Add step/2 for inclusive end
                elif isinstance(g_config, list):
                    # Direct list of values
                    self.parameter_ranges['G'] = np.array(g_config)
            
            # Extract other parameters similarly
            for param_name, param_config in grid_config.items():
                if param_name != 'G' and isinstance(param_config, dict):
                    if all(key in param_config for key in ['start', 'end', 'step']):
                        start, end, step = param_config['start'], param_config['end'], param_config['step']
                        self.parameter_ranges[param_name] = np.arange(start, end + step/2, step)
                elif isinstance(param_config, list):
                    self.parameter_ranges[param_name] = np.array(param_config)
    
    def plot_identifiability_gradient(self, 
                                    identifiability_scores: np.ndarray,
                                    parameter: str = 'G',
                                    title: Optional[str] = None,
                                    xlabel: Optional[str] = None,
                                    ylabel: str = "Identifiability Score",
                                    save_path: Optional[str] = None,
                                    show_plot: bool = True,
                                    **kwargs) -> plt.Figure:
        """
        Plot identifiability gradient with configuration-aware labeling.
        
        Parameters:
            identifiability_scores (np.ndarray): Array of identifiability scores
            parameter (str): Parameter name to use for x-axis labels (default: 'G')
            title (str, optional): Plot title
            xlabel (str, optional): X-axis label
            ylabel (str): Y-axis label
            save_path (str, optional): Path to save the plot
            show_plot (bool): Whether to display the plot
            **kwargs: Additional arguments passed to plt.plot()
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        # Get parameter values for x-axis
        if parameter in self.parameter_ranges:
            x_values = self.parameter_ranges[parameter]
            # Handle case where identifiability_scores is shorter (n-1 for gradient)
            if len(identifiability_scores) == len(x_values) - 1:
                x_values = x_values[:-1]  # Remove last value
            elif len(identifiability_scores) != len(x_values):
                warnings.warn(f"Length mismatch: identifiability_scores ({len(identifiability_scores)}) "
                            f"vs {parameter} values ({len(x_values)}). Using indices instead.")
                x_values = np.arange(len(identifiability_scores))
        else:
            x_values = np.arange(len(identifiability_scores))
            
        # Set default styling
        plot_kwargs = {
            'color': self.colors['identifiability'],
            'linewidth': 2.5,
            'marker': 'o',
            'markersize': 6,
            'markerfacecolor': 'white',
            'markeredgewidth': 2
        }
        plot_kwargs.update(kwargs)
        
        # Create the plot
        ax.plot(x_values, identifiability_scores, **plot_kwargs)
        
        # Set custom tick labels if using parameter values, selecting every 10th value
        if parameter in self.parameter_ranges:
            step = 20
            ticks = x_values[::step]
            labels = [f"{val:.2f}" for val in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        
        # Set labels and title
        ax.set_xlabel(xlabel or f"{parameter} Values" if parameter in self.parameter_ranges else "Index")
        ax.set_ylabel(ylabel)
        
        if title is None:
            title = f"Identifiability Gradient"
            if self.config.get('experiment', {}).get('name'):
                title += f" - {self.config['experiment']['name']}"
        ax.set_title(title)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_plot:
            plt.show()
            
        return fig
    
    def get_parameter_values(self, parameter: str) -> Optional[np.ndarray]:
        """
        Get parameter values from configuration.
        
        Parameters:
            parameter (str): Parameter name
            
        Returns:
            np.ndarray or None: Parameter values if found, None otherwise
        """
        return self.parameter_ranges.get(parameter)
    
    def list_available_parameters(self) -> List[str]:
        """
        List all available parameters from configuration.
        
        Returns:
            List[str]: List of parameter names
        """
        return list(self.parameter_ranges.keys())
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get experiment information from configuration.
        
        Returns:
            Dict[str, Any]: Experiment information
        """
        return self.config.get('experiment', {})
