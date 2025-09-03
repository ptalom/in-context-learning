import numpy as np
import torch


from sparse_recovery.compressed_sensing import create_signal, create_Fourier_basis, create_normal_basis, create_orthonormal_basis, get_measures

from sparse_recovery.matrix_factorization import get_matrices_UV
from sparse_recovery.matrix_factorization import get_data_matrix_factorization, solve_matrix_factorization_nuclear_norm

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

        #print(f"[Compressed Sensing Sampler] Using Phi of shape {self.Phi.shape} (type: {type(Phi).__name__})")

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

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        w_star = torch.tensor(w_star.T, dtype=torch.float32)  # (1,d)
        a_star = torch.tensor(a_star.T, dtype=torch.float32)  # (1,d)

        return X, y, w_star, a_star

class MatrixFactorizationSampler:
    
    def __init__(
        self,
        N: int,
        n1: int,
        n2: int,
        rank: int,
        problem: str = "matrix-completion",   # 'matrix-completion' / 'matrix-sensing'
        tau: float = 0.0,                     
        variance=None,
        seed: int | None = None,
        device: str = "cpu",
    ):
        assert problem in ("matrix-completion", "matrix-sensing"), \
            f"problem must be 'matrix-completion' or 'matrix-sensing', received :{problem}"
        assert 0.0 <= tau <= 1.0, f"tau must be in [0,1], received {tau}"

        self.N = N
        self.n1 = n1
        self.n2 = n2
        self.rank = rank
        self.problem = problem
        self.tau = tau
        self.variance = variance
        self.seed = seed
        self.device = device

        A_star, U_star, Sigma_star, V_star = get_matrices_UV(
            n_1=n1, n_2=n2, rank=rank, seed=seed
        )
        self.A_star = A_star.astype(np.float32)
        self.U_star = U_star.astype(np.float32)
        self.V_star = V_star.astype(np.float32)
        
        self.Sigma_star = Sigma_star.astype(np.float32)

        print(f"[Matrix Factorization Sampler] A*: {self.A_star.shape}, U*: {self.U_star.shape}, V*: {self.V_star.shape}")
        print(f"[Matrix Factorization Sampler] problem={self.problem}, N={self.N}, tau={self.tau}")

    def sample(self):
        
        # Mesures (X1, X2, X2•X1, y)
        (X1, X2, X2_bullet_X1, y_star), _ = get_data_matrix_factorization(
            self.A_star, self.U_star, self.V_star,
            self.N,
            problem=self.problem,
            tau=self.tau,
            variance=self.variance,
            seed=self.seed,
        )
        # X1: (N, n1)  | X2: (N, n2)  | y_star: (N,)

        X_np = np.concatenate([X1, X2], axis=1).astype(np.float32)
        y_np = y_star.astype(np.float32).reshape(-1)

        X = torch.tensor(X_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, N, n1+n2)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, N)

        return X, y, None, None
    


def get_data_sampler(name, **kwargs):
    name = name.lower()
    if name in ("sparse_recovery"):
        return CompressedSensingSampler(**kwargs)
    if name in ("matrix_factorization"):
        return MatrixFactorizationSampler(**kwargs)
    raise ValueError(f"Unknown sampler name: {name}")
