import torch
from torch import nn, optim
from tqdm import tqdm

from tasks import get_task
from models import build_model
import yaml
import os

def train_step(model, xs, ys, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    output = model(xs, ys)           
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.item(), output.detach()



def train(conf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modèle
    model = build_model(conf['model'])
    model = model.to(device)
    model.train()

    # Sampler et task
    data_conf = conf['task']['kwargs']
    task = get_task(conf['task']['name'], **data_conf)

    optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
    loss_func = nn.MSELoss()

    train_steps = conf['training']['train_steps']
    print_every = conf['training'].get('print_every', 10)
    save_every = conf['training'].get('save_every_steps', 100)
    out_dir = conf.get('out_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Affichage des hyperparamètres ---
    print("\n=== Training Hyperparameters ===")
    for section in ['task', 'model', 'training']:
        print(f"[{section}]")
        section_conf = conf[section]

        # Si c'est une liste de dicts
        if isinstance(section_conf, list):
            for sub_conf in section_conf:
                for k, v in sub_conf.items():
                    print(f"{k}: {v}")
        # Si c'est directement un dict
        elif isinstance(section_conf, dict):
            for k, v in section_conf.items():
                print(f"{k}: {v}")

    print(f"\n[out_dir]\n{conf['out_dir']}")
    print("================================\n")
    
    model.train()
    
    for step in tqdm(range(1, train_steps + 1)):
        xs, ys, _, _ = task.sample()  # xs: (1,N,d), ys: (1,N)
        xs = xs.to(device)
        ys = ys.to(device)
        
        loss, output = train_step(model, xs, ys, optimizer, loss_func)
        
        if step % print_every == 0 or step == 1:
            weight_mean = model._read_in.weight.mean().item()
            weight_std = model._read_in.weight.std().item()
            print(f"Step {step}/{train_steps} | Loss: {loss:.6f} | "
                  f"Weights mean: {weight_mean:.6f}, std: {weight_std:.6f}")
            print(f"Preds mean: {output.mean().item():.4f}, min: {output.min().item():.4f}, max: {output.max().item():.4f}\n")
        
        # --- Sauvegarde périodique ---
        if step % save_every == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": step
            }
            torch.save(state, os.path.join(out_dir, f"state_step_{step}.pt"))

# --- Main ---
if __name__ == "__main__":
    conf_path = "conf/compressed_sensing.yaml"
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)

    train(conf)
