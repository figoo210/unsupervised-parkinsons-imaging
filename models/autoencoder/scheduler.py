from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import math

def create_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    return ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    """Create a cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr, factor)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

