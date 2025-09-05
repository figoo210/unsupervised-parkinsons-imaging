import torch.optim as optim


def create_optimizer(model, config):
    """Create optimizer with weight decay and parameter grouping"""
    # Separate parameters that should have weight decay from those that shouldn't
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups, lr=config.learning_rate)
    
    return optimizer
