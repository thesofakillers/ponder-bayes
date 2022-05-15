import torchmetrics.functional as MF
import numpy as np


def get_model_accuracy(model, data_loader, device):
    """
    Computes average accuracy of a model instance
    on a given dataset

    Parameters
    ----------
    model : nn.Module Model instance to evaluate.
    data_loader: data.DataLoader
        The data loader of the dataset to evaluate on.
    device : torch.device
        Device to use for training.

    Returns
    -------
    accuracy : float
        The average accuracy on the dataset.
    """
    n_batches = len(data_loader)
    accuracies = np.zeros(n_batches)
    for i, (features_X, target) in enumerate(data_loader):
        features_X, target = features_X.to(device), target.to(device)
        predictions = model.forward(features_X)
        accuracies[i] = MF.accuracy(predictions, target)
    accuracy = accuracies.mean()
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


"""
TODO:
1. thinking time efficiency (ponder steps at evaluation)
2. time for training
"""
