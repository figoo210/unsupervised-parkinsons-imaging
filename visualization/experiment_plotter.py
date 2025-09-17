import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import os
from utils.logger import get_logger

# Initialize logger
logger = get_logger("experiment_plotter")

class ExperimentPlotter:
    """
    Utility class for plotting experiment results and comparisons.
    """
    def __init__(self, output_dir="output/Visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, history_df, title="Training Curves", save_path=None):
        """
        Plot training and validation loss curves.
        
        Args:
            history_df: DataFrame with training history
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training and validation loss
        axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot epoch time
        axes[1].plot(history_df["epoch"], history_df["epoch_time"])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Time (seconds)")
        axes[1].set_title("Epoch Training Time")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def plot_experiment_comparison(self, experiment_dirs, metric="val_loss", 
                                  title="Experiment Comparison", save_path=None):
        """
        Compare multiple experiments on a specific metric.
        
        Args:
            experiment_dirs: List of experiment directory paths
            metric: Metric to compare (e.g., "val_loss", "train_loss")
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        
        for exp_dir in experiment_dirs:
            exp_dir = Path(exp_dir)
            
            # Load metadata
            try:
                with open(exp_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                # Load training history
                history_df = pd.read_csv(exp_dir / "training_history.csv")
                
                # Plot the specified metric
                if metric in history_df.columns:
                    plt.plot(history_df["epoch"], history_df[metric], 
                            label=f"{metadata['experiment_name']}")
                else:
                    logger.warning(f"Metric {metric} not found in {exp_dir}")
            except Exception as e:
                logger.error(f"Error loading experiment {exp_dir}: {str(e)}")
        
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return plt.gcf()
    
    def plot_parameter_study(self, results_dict, param_name, metric_name,
                            title=None, xlabel=None, ylabel=None, 
                            log_x=False, save_path=None):
        """
        Plot the results of a parameter study.
        
        Args:
            results_dict: Dictionary of results with parameter values as keys
            param_name: Name of the parameter being studied
            metric_name: Name of the metric to plot
            title: Title for the figure
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            log_x: Whether to use log scale for x-axis
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Extract parameter values and metrics
        param_values = sorted([float(p) for p in results_dict.keys()])
        metrics = [results_dict[str(p)][metric_name] for p in param_values]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metrics, 'o-')
        
        # Set labels and title
        plt.xlabel(xlabel or param_name)
        plt.ylabel(ylabel or metric_name.replace("_", " ").title())
        plt.title(title or f"{metric_name.replace('_', ' ').title()} vs. {param_name}")
        
        # Set log scale if requested
        if log_x:
            plt.xscale("log")
        
        plt.grid(True, alpha=0.3)
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return plt.gcf()
    
    def plot_model_comparison_bar(self, results_dict, metric_names,
                                 title="Model Comparison", save_path=None):
        """
        Create a bar chart comparing multiple models on multiple metrics.
        
        Args:
            results_dict: Dictionary with model names as keys and results as values
            metric_names: List of metric names to compare
            title: Title for the figure
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Extract model names and metrics
        model_names = list(results_dict.keys())
        num_models = len(model_names)
        num_metrics = len(metric_names)
        
        # Create figure
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
        
        # Make axes iterable if there's only one metric
        if num_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            
            # Extract metric values for each model
            metric_values = [results_dict[model][metric] for model in model_names]
            
            # Create bar chart
            ax.bar(model_names, metric_values)
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels(model_names, rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def plot_parameter_vs_metrics(self, results_dict, param_name, metric_names,
                                 title=None, xlabel=None, log_x=False, save_path=None):
        """
        Plot multiple metrics against a parameter.
        
        Args:
            results_dict: Dictionary of results with parameter values as keys
            param_name: Name of the parameter being studied
            metric_names: List of metric names to plot
            title: Title for the figure
            xlabel: Label for x-axis
            log_x: Whether to use log scale for x-axis
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        # Extract parameter values
        param_values = sorted([float(p) for p in results_dict.keys()])
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot each metric
        for metric in metric_names:
            metrics = [results_dict[str(p)][metric] for p in param_values]
            plt.plot(param_values, metrics, 'o-', label=metric.replace("_", " ").title())
        
        # Set labels and title
        plt.xlabel(xlabel or param_name)
        plt.ylabel("Value")
        plt.title(title or f"Metrics vs. {param_name}")
        
        # Set log scale if requested
        if log_x:
            plt.xscale("log")
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure if save_path is provided
        if save_path:
            if not isinstance(save_path, Path):
                save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return plt.gcf()
    
    def create_summary_report(self, experiment_dir, output_path=None):
        """
        Create a summary report for an experiment.
        
        Args:
            experiment_dir: Directory containing experiment results
            output_path: Path to save the summary report
        
        Returns:
            Summary report as a string
        """
        experiment_dir = Path(experiment_dir)
        
        # Load metadata
        try:
            with open(experiment_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load training history
            history_df = pd.read_csv(experiment_dir / "training_history.csv")
            
            # Create summary report
            summary = []
            summary.append(f"# Experiment Summary: {metadata['experiment_name']}")
            summary.append(f"Timestamp: {metadata['timestamp']}")
            summary.append(f"Device: {metadata['device']}")
            summary.append("")
            
            # Model information
            summary.append("## Model Information")
            summary.append(f"Total Parameters: {metadata.get('total_parameters', 'N/A'):,}")
            summary.append(f"Trainable Parameters: {metadata.get('trainable_parameters', 'N/A'):,}")
            summary.append(f"FLOPs: {metadata.get('flops', 'N/A')}")
            summary.append("")
            
            # Training results
            summary.append("## Training Results")
            summary.append(f"Epochs: {len(history_df)}")
            summary.append(f"Final Training Loss: {history_df['train_loss'].iloc[-1]:.6f}")
            summary.append(f"Final Validation Loss: {history_df['val_loss'].iloc[-1]:.6f}")
            summary.append(f"Best Validation Loss: {history_df['val_loss'].min():.6f} (Epoch {history_df['val_loss'].idxmin()})")
            summary.append(f"Average Epoch Time: {history_df['epoch_time'].mean():.2f} seconds")
            summary.append("")
            
            # Additional metrics if available
            if 'train_recon_loss' in history_df.columns:
                summary.append("## Additional Metrics")
                summary.append(f"Final Training Reconstruction Loss: {history_df['train_recon_loss'].iloc[-1]:.6f}")
                summary.append(f"Final Validation Reconstruction Loss: {history_df['val_recon_loss'].iloc[-1]:.6f}")
                summary.append("")
            
            # Join summary into a single string
            summary_text = "\n".join(summary)
            
            # Save summary if output_path is provided
            if output_path:
                if not isinstance(output_path, Path):
                    output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(summary_text)
                logger.info(f"Saved summary report to {output_path}")
            
            return summary_text
        
        except Exception as e:
            logger.error(f"Error creating summary report for {experiment_dir}: {str(e)}")
            return None
