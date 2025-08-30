
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model   

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
