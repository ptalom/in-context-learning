import os
import json
from munch import Munch
import torch
import yaml
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

import models

from models import build_model
from samplers import get_data_sampler
from samplers import CompressedSensingSampler, MatrixFactorizationSampler
from tasks import get_task


# --- Chargement du modèle depuis un run ---
def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    
    if only_conf:
        return None, conf

    model = build_model(conf.model)

    checkpoints_dir = os.path.join(run_path, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        print(f"Warning: {checkpoints_dir} not found. Using randomly initialized model.")
        return model, conf

    if step == -1:
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "model_step_*.pt"))
        if not checkpoint_files:
            print(f"Warning: aucun checkpoint trouvé dans {checkpoints_dir}. Utilisation d'un modèle aléatoire.")
            return model, conf
        steps = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in checkpoint_files]
        latest_step = max(steps)
        model_path = os.path.join(checkpoints_dir, f"model_step_{latest_step}.pt")
    else:
        model_path = os.path.join(checkpoints_dir, f"model_step_{step}.pt")

    # Charger le checkpoint complet
    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint  

    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    
    return model, conf


def eval_batch(model, task_sampler, xs, xs_p=None):
    """
    Évalue le modèle sur un batch, avec ou sans permutation des points.

    Args:
        model : le modèle à évaluer
        task_sampler : l'objet task qui permet de générer les y et les métriques
        xs : Tensor (batch_size, N, d) ou (batch_size, N)
        xs_p : Tensor optionnel, same shape as xs, pour la permutation des points

    Returns:
        metrics : Tensor (batch_size, N)
    """
    device = "cuda" if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"] else "cpu"

    # S'assurer que xs a 3 dimensions
    if xs.ndim == 2:
        xs = xs.unsqueeze(-1)  # [batch, N, 1]

    if xs_p is not None and xs_p.ndim == 2:
        xs_p = xs_p.unsqueeze(-1)

    # S'assurer que xs_p a le même dernier dimension que xs
    if xs_p is not None and xs_p.shape[-1] != xs.shape[-1]:
        xs_p = xs_p.expand(-1, -1, xs.shape[-1])

    if xs_p is None:
        ys = task_sampler.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task_sampler.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, dim = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            # Découpage sécurisé des deux parties
            xs_left = xs[:, :i, :] if i > 0 else torch.zeros(b_size, 0, dim)
            xs_right = xs_p[:, i:, :] if i < n_points else torch.zeros(b_size, 0, dim)

            # Concat sur la dimension des points (1)
            xs_comb = torch.cat((xs_left, xs_right), dim=1)

            ys = task_sampler.evaluate(xs_comb)
            pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            metrics[:, i] = task_sampler.get_metric()(pred.cpu(), ys)

    return metrics



def generate_eval_data(task_name, N, tau, **kwargs):
    """
    Wrapper pour générer un batch d'évaluation, selon la tâche.
    Pour sparse_recovery : kwargs = d, s, Phi
    Pour matrix_factorization : kwargs = n1, n2, rank, problem
        problem ∈ {"matrix-completion", "matrix-sensing"}
    """
    if task_name == "sparse_recovery":
        required = ["d", "s", "Phi"]
        for r in required:
            if r not in kwargs:
                raise ValueError(f"Pour sparse_recovery, il faut fournir {required}")
        sampler = get_data_sampler(
            "sparse_recovery",
            N=N,
            d=kwargs["d"],
            s=kwargs["s"],
            tau=tau,
            Phi=kwargs["Phi"]
        )

    elif task_name == "matrix_factorization":
        required = ["n1", "n2", "rank", "problem"]
        for r in required:
            if r not in kwargs:
                raise ValueError(f"Pour matrix_factorization, il faut fournir {required}")
        sampler = get_data_sampler(
            "matrix_factorization",
            N=N,
            n1=kwargs["n1"],
            n2=kwargs["n2"],
            rank=kwargs["rank"],
            tau=tau,
            problem=kwargs["problem"]  
        )

    else:
        raise ValueError(f"Tâche inconnue : {task_name}")

    return sampler.sample()


# --- Evaluation complète ---
def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs=None,
    task_sampler_kwargs=None,
):
    """
    Evaluate a model on a task with a variety of strategies.
    """

    assert num_eval_examples % batch_size == 0

    data_sampler_kwargs = {} if data_sampler_kwargs is None else data_sampler_kwargs
    task_sampler_kwargs = {} if task_sampler_kwargs is None else task_sampler_kwargs

    
    valid_data_keys = ["N", "d", "s", "Phi", "tau"]
    filtered_data_kwargs = {k: v for k, v in data_sampler_kwargs.items() if k in valid_data_keys}

    data_sampler = CompressedSensingSampler(**filtered_data_kwargs)

    task_sampler_kwargs = {} if task_sampler_kwargs is None else task_sampler_kwargs
    task_sampler_kwargs.update(filtered_data_kwargs)  # N, d, s, Phi, tau
    task_sampler = get_task(task_name, **task_sampler_kwargs)

    all_metrics = []

    for i in range(num_eval_examples // batch_size):
        xs, xs_p, _, _ = generate_eval_data(
            task_name,
            N=data_sampler_kwargs.get("N"),
            tau=data_sampler_kwargs.get("tau", 1),
            **{k: v for k, v in data_sampler_kwargs.items() if k not in ["N", "tau"]}
        )
        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)
    metrics = torch.cat(all_metrics, dim=0)
    return aggregate_metrics(metrics)

def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    if bootstrap_means.dim() == 1:
        bootstrap_means = bootstrap_means.unsqueeze(1)  # devient (T,1)

    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        setattr(model, "name", conf['model'].get('type', 'Transformer'))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.task.name)

    # --- Préparer les kwargs pour eval_model ---
    # Inclut les paramètres de task et du dataset
    data_sampler_kwargs = conf.task.kwargs
    data_sampler_kwargs["N"] = data_sampler_kwargs.get("N", conf.model.n_positions)
    data_sampler_kwargs["d"] = data_sampler_kwargs.get("d", conf.model.n_dims)
    # Créer evaluation_kwargs sous forme de dictionnaire
    evaluation_kwargs = {
        "default_eval": {
            "task_name": conf.task.name,
            "data_name": "sparse_recovery",
            "n_dims": conf.model.n_dims,
            "n_points": conf.model.n_positions,
            "prompting_strategy": "random",
            "data_sampler_kwargs": {
                "N": conf.task.kwargs.get("N"),
                "d": conf.task.kwargs.get("d"),
                "s": conf.task.kwargs.get("s"),
                "Phi": conf.task.kwargs.get("Phi", "normal"),
                "tau": conf.task.kwargs.get("tau", 1),
            },
        }
    }


    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        # Extraire les paramètres nécessaires pour eval_model
        task_name = kwargs.get("task_name", "sparse_recovery")
        data_name = kwargs.get("data_name", "compressed_sensing")
        n_dims = kwargs.get("n_dims")
        n_points = kwargs.get("n_points")
        prompting_strategy = kwargs.get("prompting_strategy", "random")
        data_sampler_kwargs = kwargs.get("data_sampler_kwargs", {})

        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]

        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            # Appel à eval_model avec les bons arguments
            metrics[model.name] = eval_model(
                model,
                task_name=task_name,
                data_name=data_name,
                n_dims=n_dims,
                n_points=n_points,
                prompting_strategy=prompting_strategy,
                data_sampler_kwargs=data_sampler_kwargs
            )

        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics



def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    return name

def read_run_dir(run_dir):
    all_data = []

    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        if not os.path.isdir(task_dir):
            continue
        
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            if not os.path.isdir(run_path):
                continue
            
            config_path = os.path.join(run_path, "config.yaml")
            if not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                conf = Munch.fromDict(yaml.safe_load(f))

            run_dict = {
                "run_id": run_id,
                "model": conf.model.type,
                "task": conf.task.name,
                "run_path": run_path,   # 🔹 ajouter run_path ici
            }
            all_data.append(run_dict)

    df = pd.DataFrame(all_data)
    return df



# --- Boucle principale ---
if __name__ == "__main__":
    run_dir = "outputs"
    df_runs = read_run_dir(run_dir)
    all_results = []

    for _, row in df_runs.iterrows():
        task_name = row['task']
        print(f"=== Evaluating {task_name} / {row['run_id']} ===")
    
        model, conf = get_model_from_run(row['run_path'])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        
        if task_name == "sparse_recovery":
            taus = [0.0, 0.5, 1.0]
            results = eval_model(model, task_name, N=16, taus=taus, d=conf.model.n_dims, s=5, Phi=torch.eye(conf.model.n_dims))
        elif task_name == "matrix_factorization":
            taus = [0.0, 0.5, 1.0]
            results = eval_model(model, task_name, N=16, taus=taus, n1=20, n2=20, rank=5)
        
        # Transformer les résultats en liste de dicts pour Pandas
        for tau, metric in results.items():
            all_results.append({
                "task": task_name,
                "run_id": row['run_id'],
                "tau": tau,
                "metric": metric.mean().item() if isinstance(metric, torch.Tensor) else metric
            })

        # Créer le DataFrame final
        results_df = pd.DataFrame(all_results)
        print("\n=== Summary ===")
        print(results_df)

    # Sauvegarder en JSON pour analyse future
    results_df.to_json(os.path.join(run_dir, "all_runs_metrics.json"), orient="records", indent=2)
    print("\nMetrics saved to all_runs_metrics.json")
