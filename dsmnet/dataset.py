from torch.utils.data import Dataset
import numpy as np

class RSDataset(Dataset):
    """Seismic record section dataset.

    Samples are tuples (record section, model perturbation)

    """

    def __init__(self, path_x, path_y):
        self.samples = []

        with open(path_x, 'rb') as f:
            X = np.load(f)

        with open(path_y, 'rb') as f:
            y = np.load(f)

        X = X.transpose((0, 1, 3, 2))
        X = _normalize_X(X)
        y = _normalize_Y(y)

        for i in range(X.shape[0]):
            self.samples.append((X[i], y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _normalize_X(X):
    maxs = np.abs(X).max(axis=3, keepdims=True)
    X /= maxs
    return X

def _normalize_Y(Y):
    maxs = np.array([0.5, 190.]).reshape(1, 2)
    return Y / maxs

