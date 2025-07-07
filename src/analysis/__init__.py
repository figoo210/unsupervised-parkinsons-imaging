"""
Analysis and visualization functions for the medical image analysis project.
"""

from .visualization import (
    plot_training_history, 
    plot_vae_training_history,
    visualize_reconstruction_samples,
    visualize_vae_reconstructions,
    visualize_latent_space,
    plot_latent_dimension_activation,
    visualize_vae_uncertainty
)

from .evaluation import (
    compute_reconstruction_error,
    evaluate_model_performance,
    find_outliers,
    calculate_metrics
)

from .interpretation import (
    extract_latent_vectors,
    generate_feature_importance_map,
    analyze_latent_dimensions,
    visualize_latent_dimension
)

__all__ = [
    # Visualization functions
    'plot_training_history',
    'plot_vae_training_history', 
    'visualize_reconstruction_samples',
    'visualize_vae_reconstructions',
    'visualize_latent_space',
    'plot_latent_dimension_activation',
    'visualize_vae_uncertainty',
    
    # Evaluation functions
    'compute_reconstruction_error',
    'evaluate_model_performance',
    'find_outliers',
    'calculate_metrics',
    
    # Interpretation functions
    'extract_latent_vectors',
    'generate_feature_importance_map',
    'analyze_latent_dimensions',
    'visualize_latent_dimension'
]
