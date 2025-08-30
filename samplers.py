import numpy as np
import torch
from sparse_recovery.compressed_sensing import create_signal, create_Fourier_basis, create_normal_basis, create_orthonormal_basis, get_measures

from sparse_recovery.matrix_factorization import get_matrices_UV, get_matrices_IndentityUV, get_matrices_QR_UV, calculate_local_coherence
from sparse_recovery.matrix_factorization import get_data_matrix_factorization, solve_matrix_factorization_nuclear_norm
from sparse_recovery.matrix_factorization import get_measures_matrix_completion

class CompressedSensingSampler:
    def __init__(self, N, d, s, Phi=None, tau=0, variance=None, seed=None):
        
        self.N = N
        self.d = d
        self.s = s
        self.tau = tau
        self.variance = variance
        self.seed = seed
        if not (0.0 <= self.tau <= 1.0):
          raise ValueError(f"tau must in [0,1], received {self.tau}")

        if Phi is None or Phi == "identity":
            self.Phi = np.eye(d, dtype=np.float32)
        elif isinstance(Phi, str):
            Phi = Phi.lower()
            if Phi == "fourier":
                self.Phi = np.array(create_Fourier_basis(d), dtype=np.float32)
            elif Phi == "normal":
                self.Phi = np.array(create_normal_basis(d, seed=seed), dtype=np.float32)
            elif Phi == "orthonormal":
                self.Phi = np.array(create_orthonormal_basis(d, seed=seed), dtype=np.float32)
            else:
                raise ValueError(f"Unknown Phi type: {Phi}")
        else:
            self.Phi = np.array(Phi, dtype=np.float32)

        print(f"[Sampler] Using Phi of shape {self.Phi.shape} (type: {type(Phi).__name__})")

    def sample(self):
        """
        xs : (N, d)
        ys : (N,)
        """
        # Génération du signal sparse a*
        a_star, _ = create_signal(n=self.d, s=self.s, distribution="normal", Phi=None, scaler=None, seed=self.seed)
        a_star = np.array(a_star, dtype=np.float32)  

        # Génération de w* = Phi @ a*
        w_star = self.Phi @ a_star

        # Génération de la matrice de mesures M
        M = get_measures(
            N=self.N,
            Phi=self.Phi,
            tau=self.tau,
            variance=self.variance,
            seed=self.seed,
        )

        # Données (X, y) : y = M @ w*
        X = np.array(M, dtype=np.float32)    # shape (N, d)
        y = X @ w_star                        # shape (N,1)

        
        # Conversion en torch
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        w_star = torch.tensor(w_star.T, dtype=torch.float32)  # (1,d)
        a_star = torch.tensor(a_star.T, dtype=torch.float32)  # (1,d)

        return X, y, w_star, a_star
    

class MatrixFactorizationSampler:
    """
    Wrapper for get_measures_matrix_completion.
    Returns Xt (1,N,feat_dim), yt (1,N)
    If one_hot=False encodes (i,j) as 2 scalars; else one-hot concatenation.
    """
    def __init__(self, A_star, N, X1=None, X2=None, P=None, tau=0.0, one_hot=False, shuffle=True, seed=None, device='cpu'):
        self.A_star = np.asarray(A_star)
        self.N = N
        self.X1 = X1
        self.X2 = X2
        self.P = P
        self.tau = tau
        self.one_hot = one_hot
        self.shuffle = shuffle
        self.seed = seed
        self.device = device

    def sample(self):
        X1, X2, y_star = get_measures_matrix_completion(self.A_star, self.N, X1=self.X1, X2=self.X2, P=self.P, tau=self.tau, one_hot=self.one_hot, shuffle=self.shuffle, seed=self.seed)
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y_star).reshape(-1)
        if self.one_hot:
            n1, n2 = self.A_star.shape
            enc = np.concatenate([np.eye(n1)[X1.astype(int)], np.eye(n2)[X2.astype(int)]], axis=1)
        else:
            enc = np.stack([X1.astype(float), X2.astype(float)], axis=1)  # (N,2)
        Xt = torch.tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,N,feat)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)    # (1,N)
        return Xt, yt

def get_data_sampler(name, **kwargs):
    name = name.lower()
    if name in ("sparse_recovery"):
        return CompressedSensingSampler(**kwargs)
    if name in ("matrix_factorization"):
        return MatrixFactorizationSampler(**kwargs)
    raise ValueError(f"Unknown sampler name: {name}")
