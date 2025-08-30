from samplers import get_data_sampler
import torch

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)

def mse_metric(pred, target):
    return ((pred - target) ** 2).mean().item()

def get_task(task_name, **kwargs):
    """
    Returns an object with .sample() -> (X_torch, y_torch) and .metric(pred, target)
    For sparse_recovery: kwargs should contain N, Phi, a_star or w_star, tau, ...
    For matrix_factorization: kwargs should contain A_star, N, one_hot, tau, ...
    """
    sampler = get_data_sampler(task_name, **kwargs)
    class SimpleTask:
        def __init__(self, sampler):
            self.sampler = sampler
        def sample(self):
            return self.sampler.sample()
        def metric(self, pred, target):
            return mse_metric(pred, target)
    return SimpleTask(sampler)
