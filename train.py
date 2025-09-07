import torch
from torch import nn, optim
from random import randint
from tqdm import tqdm
from munch import Munch
import yaml
import os
import argparse
import json
from datetime import datetime
import sys

from samplers import get_data_sampler
from tasks import get_task
from models import build_model
from eval import get_run_metrics  # Import de la fonction pour calculer les métriques


torch.backends.cudnn.benchmark = True

def train_step(model, xs, ys, optimizer, loss_func):
    model.train()
    optimizer.zero_grad()
    output = model(xs, ys)           
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.item(), output.detach()

def create_run_dir(base_dir="outputs", task="default_task", config=None):
    #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #run_dir = os.path.join(base_dir, task, timestamp)
    run_dir = os.path.join(base_dir, task, "model_pretrained")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    if config is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.dump(config.toDict() if isinstance(config, Munch) else config, f)

    return run_dir


def save_checkpoint(run_dir, model_state, optimizer_state, step):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"model_step_{step}.pt")
    torch.save({
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "step": step
    }, ckpt_path)

def train(conf, recompute_metrics=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer dossier de run
    task_name = conf['task']['name']
    run_dir = create_run_dir(base_dir="outputs", task=task_name, config=conf)

    # Modèle
    model = build_model(conf['model'])
    setattr(model, "name", conf['model'].get('type', 'Transformer'))    
    model = model.to(device)
    model.train()

    # Sampler et task
    data_conf = conf['task']['kwargs']
    task = get_task(task_name, **data_conf)

    optimizer = optim.Adam(model.parameters(), lr=conf['training']['learning_rate'])
    loss_func = nn.MSELoss()

    train_steps = conf['training']['train_steps']
    print_every = conf['training'].get('print_every', 10)
    save_every = conf['training'].get('save_every_steps', 100)

    # --- Affichage des hyperparamètres ---
    print("\n=== Training Hyperparameters ===")
    for section in ['task', 'model', 'training']:
        print(f"[{section}]")
        section_conf = conf[section]
        if isinstance(section_conf, list):
            for sub_conf in section_conf:
                for k, v in sub_conf.items():
                    print(f"{k}: {v}")
        elif isinstance(section_conf, dict):
            for k, v in section_conf.items():
                print(f"{k}: {v}")
    print("================================\n")

    # --- Boucle d'entraînement ---
    for step in tqdm(range(1, train_steps + 1)):
        xs, ys, _, _ = task.sample()
        xs, ys = xs.to(device), ys.to(device)
        # Flatten ys pour matcher output
        ys = ys.view(ys.size(0), -1)   # [1, 100]

        loss, output = train_step(model, xs, ys, optimizer, loss_func)
        
        if step % print_every == 0 or step == 1:
            weight_mean = model._read_in.weight.mean().item()
            weight_std = model._read_in.weight.std().item()
            print(f"Step {step}/{train_steps} | Loss: {loss:.6f} | "
                  f"Weights mean: {weight_mean:.6f}, std: {weight_std:.6f}")
            print(f"Preds mean: {output.mean().item():.4f}, "
                  f"min: {output.min().item():.4f}, max: {output.max().item():.4f}\n")

        if step % save_every == 0:
            save_checkpoint(run_dir, model.state_dict(), optimizer.state_dict(), step)


    # --- Calcul automatique des métriques à la fin ---
    if recompute_metrics:
        print("\n=== Computing run metrics ===")
        get_run_metrics(run_dir)
        print(get_run_metrics(run_dir))

# --- Main ---
'''
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/compressed_sensing.yaml',
                        help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        conf_dict = yaml.safe_load(f)

    conf = Munch.fromDict(conf_dict)

    train(conf, recompute_metrics=True)
