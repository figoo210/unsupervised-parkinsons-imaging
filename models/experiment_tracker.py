import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

# Initialize logger
logger = get_logger("experiment_tracker")

class ExperimentTracker:
    """
    Tracks and records experiment metrics, parameters, and results.
    Provides utilities for saving and loading experiment data.
    """
    def __init__(self, experiment_name, base_dir="output/Experiments"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        
        # Create experiment directory
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment data
        self.metadata = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "pytorch_version": torch.__version__,
            "experiment_id": self.experiment_id
        }
        
        self.model_info = {}
        self.training_history = {
            "train_losses": [],
            "val_losses": [],
            "train_recon_losses": [],
            "val_recon_losses": [],
            "epochs": [],
            "epoch_times": []
        }
        
        self.reconstruction_samples = {}
        
        logger.info(f"Initialized experiment tracker: {self.experiment_id}")
    
    def record_model_info(self, model, input_shape=(1, 64, 128, 128)):
        """Record model architecture details and parameter count"""
        device = next(model.parameters()).device
        
        # Get parameter count
        param_count = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model architecture summary
        model_str = str(model)
        
        # Try to get FLOPs if possible (requires additional libraries)
        flops = "Not calculated"
        try:
            from thop import profile
            dummy_input = torch.randn(1, *input_shape).to(device)
            flops, _ = profile(model, inputs=(dummy_input,))
            flops = f"{flops / 1e9:.2f} GFLOPs"
        except ImportError:
            logger.warning("thop library not found, FLOPs not calculated")
        except Exception as e:
            logger.warning(f"Error calculating FLOPs: {str(e)}")
        
        self.model_info = {
            "architecture": model_str,
            "total_parameters": param_count,
            "trainable_parameters": trainable_param_count,
            "flops": flops,
            "input_shape": str(input_shape)
        }
        
        logger.info(f"Recorded model info: {param_count:,} parameters ({trainable_param_count:,} trainable)")
        return self.model_info
    
    def record_epoch(self, epoch, train_loss, val_loss, epoch_time, 
                    train_recon_loss=None, val_recon_loss=None):
        """Record metrics for a training epoch"""
        self.training_history["epochs"].append(epoch)
        self.training_history["train_losses"].append(float(train_loss))
        self.training_history["val_losses"].append(float(val_loss))
        self.training_history["epoch_times"].append(float(epoch_time))
        
        if train_recon_loss is not None:
            self.training_history["train_recon_losses"].append(float(train_recon_loss))
        if val_recon_loss is not None:
            self.training_history["val_recon_losses"].append(float(val_recon_loss))
        
        logger.debug(f"Recorded epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    def record_reconstruction_samples(self, original, reconstructed, epoch, num_samples=3):
        """Record sample reconstructions for visualization"""
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Take a subset of samples
        if original.shape[0] > num_samples:
            indices = np.random.choice(original.shape[0], num_samples, replace=False)
            original = original[indices]
            reconstructed = reconstructed[indices]
        
        self.reconstruction_samples[epoch] = {
            "original": original,
            "reconstructed": reconstructed
        }
        
        logger.debug(f"Recorded {original.shape[0]} reconstruction samples for epoch {epoch}")
    
    def save_experiment(self):
        """Save all experiment data to disk"""
        # Save metadata and model info
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({**self.metadata, **self.model_info}, f, indent=2)
        
        # Save training history
        history_file = self.experiment_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save training history as CSV for easy analysis
        history_df = pd.DataFrame({
            "epoch": self.training_history["epochs"],
            "train_loss": self.training_history["train_losses"],
            "val_loss": self.training_history["val_losses"],
            "epoch_time": self.training_history["epoch_times"]
        })
        if self.training_history["train_recon_losses"]:
            history_df["train_recon_loss"] = self.training_history["train_recon_losses"]
        if self.training_history["val_recon_losses"]:
            history_df["val_recon_loss"] = self.training_history["val_recon_losses"]
        
        history_df.to_csv(self.experiment_dir / "training_history.csv", index=False)
        
        # Save reconstruction samples
        if self.reconstruction_samples:
            recon_dir = self.experiment_dir / "reconstructions"
            recon_dir.mkdir(exist_ok=True)
            
            for epoch, samples in self.reconstruction_samples.items():
                np.savez(
                    recon_dir / f"recon_samples_epoch_{epoch}.npz",
                    original=samples["original"],
                    reconstructed=samples["reconstructed"]
                )
        
        # Generate and save training curve plot
        self._save_training_curve_plot()
        
        logger.info(f"Saved experiment data to {self.experiment_dir}")
        return str(self.experiment_dir)
    
    def _save_training_curve_plot(self):
        """Generate and save training curve plot"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history["epochs"], self.training_history["train_losses"], label="Train Loss")
        plt.plot(self.training_history["epochs"], self.training_history["val_losses"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history["epochs"], self.training_history["epoch_times"])
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.title("Epoch Training Time")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "training_curves.png")
        plt.close()
    
    @staticmethod
    def load_experiment(experiment_dir):
        """Load an experiment from disk"""
        experiment_dir = Path(experiment_dir)
        
        # Load metadata
        with open(experiment_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create experiment tracker instance
        tracker = ExperimentTracker(metadata["experiment_name"])
        tracker.experiment_id = metadata["experiment_id"]
        tracker.timestamp = metadata["timestamp"]
        tracker.experiment_dir = experiment_dir
        tracker.metadata = {k: v for k, v in metadata.items() 
                          if k not in ["architecture", "total_parameters", 
                                      "trainable_parameters", "flops", "input_shape"]}
        
        # Load model info
        tracker.model_info = {k: v for k, v in metadata.items() 
                             if k in ["architecture", "total_parameters", 
                                     "trainable_parameters", "flops", "input_shape"]}
        
        # Load training history
        with open(experiment_dir / "training_history.json", 'r') as f:
            tracker.training_history = json.load(f)
        
        # Load reconstruction samples if they exist
        recon_dir = experiment_dir / "reconstructions"
        if recon_dir.exists():
            for npz_file in recon_dir.glob("recon_samples_epoch_*.npz"):
                epoch = int(npz_file.stem.split("_")[-1])
                data = np.load(npz_file)
                tracker.reconstruction_samples[epoch] = {
                    "original": data["original"],
                    "reconstructed": data["reconstructed"]
                }
        
        logger.info(f"Loaded experiment from {experiment_dir}")
        return tracker
    
    @staticmethod
    def compare_experiments(experiment_dirs, metrics=["val_loss"], figsize=(12, 6)):
        """Compare multiple experiments on specified metrics"""
        plt.figure(figsize=figsize)
        
        for exp_dir in experiment_dirs:
            try:
                # Load experiment
                tracker = ExperimentTracker.load_experiment(exp_dir)
                exp_name = tracker.experiment_name
                
                # Plot each requested metric
                for metric in metrics:
                    if metric in tracker.training_history:
                        plt.plot(
                            tracker.training_history["epochs"],
                            tracker.training_history[metric],
                            label=f"{exp_name} - {metric}"
                        )
            except Exception as e:
                logger.error(f"Error loading experiment {exp_dir}: {str(e)}")
        
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Experiment Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
