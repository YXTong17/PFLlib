import torch.nn as nn


def replace_bn_with_gn(model, num_groups=32, device="cuda"):
    """
    Replace all BatchNorm2d layers in a model with GroupNorm.

    Args:
        model (nn.Module): The model to modify.
        num_groups (int): Number of groups for GroupNorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Get the number of channels in the BN layer
            num_channels = module.num_features
            # Replace BN with GN
            setattr(
                model,
                name,
                nn.GroupNorm(
                    num_groups=num_groups, num_channels=num_channels, device=device
                ),
            )
        elif len(list(module.children())) > 0:  # Recursively replace in child modules
            replace_bn_with_gn(module, num_groups)
