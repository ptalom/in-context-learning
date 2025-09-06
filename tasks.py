import torch

from samplers import CompressedSensingSampler, MatrixFactorizationSampler


class SimpleTask:
    """Classe de base pour toutes les tâches."""
    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self, batch_size=1):
        """Retourne xs, ys, w*, a* selon le sampler."""
        return self.sampler.sample(batch_size=batch_size)

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


# --- Fonction utilitaire pour récupérer la tâche ---
def get_task(task_name, **kwargs):
    if task_name.lower() == "sparse_recovery":
        return SparseRecoveryTask(**kwargs)
    elif task_name.lower() == "matrix_factorization":
        return MatrixFactorizationTask(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}")

# --- Fonctions de métriques ---
def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


# --- Sparse Recovery Task ---
class SparseRecoveryTask(SimpleTask):
    def __init__(self, **kwargs):
        sampler = CompressedSensingSampler(**kwargs)
        super().__init__(sampler)

    def evaluate(self, xs):
        w_star = self.sampler.sample()[2]  # shape: (batch, d)
        # tronquer xs si n_features > d
        if xs.shape[2] > w_star.shape[1]:
            xs = xs[:, :, :w_star.shape[1]]
        return torch.matmul(xs, w_star.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def get_metric():
        return mean_squared_error


# --- Matrix Factorization Task ---
class MatrixFactorizationTask(SimpleTask):
    def __init__(self, **kwargs):
        sampler = MatrixFactorizationSampler(**kwargs)
        super().__init__(sampler)

    def evaluate(self, xs):
        # Ici xs n’est pas toujours utilisé, car sampler génère y directement
        return self.sampler.sample()[1]  # y

    @staticmethod
    def get_metric():
        return mean_squared_error


