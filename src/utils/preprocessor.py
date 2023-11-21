import torch
import math

def find_extrema(dataset):
    """
    Given the dataset find the global min and max value.

    Args:
        dataset (MagnaTagATune): the dataset

    Returns:
        global_min (int): minimum value in the dataset
        global_max (int): maximum value in the dataset.
    """

    global_min = math.inf
    global_max = -math.inf
    for i in range(len(dataset)):
        _, batch, _ = dataset[i]
        flattened = torch.flatten(batch)
        batch_min = torch.min(flattened)
        batch_max = torch.max(flattened)
        global_min = min(global_min, batch_min)
        global_max = max(global_max, batch_max)
    
    print(global_min, global_max)
    return global_min, global_max
