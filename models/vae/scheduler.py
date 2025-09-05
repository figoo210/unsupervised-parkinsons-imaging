from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_vae_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    return ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
