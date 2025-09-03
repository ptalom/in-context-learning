
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model   

from sklearn.linear_model import Lasso
import warnings

def build_model(conf):
    """
    Crée le TransformerModel à partir de la configuration YAML.
    conf doit contenir au minimum :
        n_dims: dimension d'entrée des x
        n_positions: longueur max des séquences
        n_embd: dimension des embeddings (optionnel)
        n_layer: nombre de couches Transformer (optionnel)
        n_head: nombre de têtes d'attention (optionnel)
    """
    n_dims = conf["n_dims"]
    n_positions = conf["n_positions"]
    n_embd = conf.get("n_embd", 128)
    n_layer = conf.get("n_layer", 4)
    n_head = conf.get("n_head", 4)

    return TransformerModel(n_dims, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "sparse_recovery": [
            (LeastSquaresModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]
        ],
        "matrix_factorization": [
            (LeastSquaresModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]
        ],

    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models



class TransformerModel(nn.Module):
    """
    Small encoder-based Transformer for ICL-style experiments.
    Signature: forward(xs, ys, inds=None)
      - xs: (batch, seq_len, n_dims)
      - ys: (batch, seq_len)          (float values)
      - inds: list/array of positions to return (relative to seq)
    Returns preds: (batch, len(inds))
    """
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=4, n_head=4):
        super().__init__()
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd

        self._read_in = nn.Linear(n_dims, n_embd)
    
        cfg = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self._backbone = GPT2Model(cfg)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleave xs and ys into sequence of length 2*L:
          (x1, y1_wide, x2, y2_wide, ..., xL, yL_wide)
        where y*_wide is a vector of same dim as x with y in first coord and zeros else.
        xs_b: (batch, L, dim)
        ys_b: (batch, L)
        => out (batch, 2L, dim)
        """
        bsize, L, dim = xs_b.shape
        ys_wide = torch.cat([ys_b.view(bsize, L, 1),
                              torch.zeros(bsize, L, dim - 1, device=xs_b.device)], dim=2)
        zs = torch.stack((xs_b, ys_wide), dim=2)           
        zs = zs.view(bsize, 2 * L, dim)                    
        return zs

    def forward(self, xs, ys, inds=None):
        """
        xs: torch.FloatTensor (batch, L, n_dims)
        ys: torch.FloatTensor (batch, L)
        inds: list indices among positions of xs we want to predict (indices in 0..L-1)
        Returns: preds (batch, len(inds))
        """

        zs = self._combine(xs, ys)            
        embeds = self._read_in(zs)            
        output = self._backbone(inputs_embeds=embeds).last_hidden_state  
        pred_all = self._read_out(output).squeeze(-1)                   

        preds_x_positions = pred_all[:, 0::2]  
        if inds is None:
            return preds_x_positions
        if isinstance(inds, int):
            inds = [inds]
        inds = torch.tensor(inds, dtype=torch.long, device=preds_x_positions.device)
        return preds_x_positions[:, inds]



class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)



class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)
