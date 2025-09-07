import os
import json
import sys

from munch import Munch
import torch
import yaml
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

import models

from models import build_model
from samplers import get_data_sampler, sample_transformation
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
    print("go eval_batch")
    return metrics


'''
def gen_coherence_tau(data_sampler, n_points, b_size, tau):
    """
    Génère un prompt (xs_train_pre, xs_test_post) pour la sparse recovery
    avec un degré de cohérence contrôlé par tau.
    """

    # Tirage initial d'exemples
    xs = data_sampler.sample(n_points, b_size)
    
    # Si xs est un tuple (xs, ...) prendre le premier élément
    if isinstance(xs, tuple):
        xs = xs[0]

    xs_train_pre = xs.clone()
    xs_test_post = xs.clone()

    # Cas tau = 0 : pure incohérence
    if tau == 0:
        xs_train_pre = torch.randn_like(xs_train_pre)

    else:
        # Convertir Phi en torch.Tensor si nécessaire
        Phi = data_sampler.Phi
        if not isinstance(Phi, torch.Tensor):
            Phi = torch.tensor(Phi, dtype=xs_train_pre.dtype, device=xs_train_pre.device)

        # SVD
        U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
        proj = U @ U.transpose(0, 1)

        if tau == 1:
            xs_train_pre = xs_train_pre @ proj
        elif tau == 0.5:
            xs_proj = xs_train_pre @ proj
            xs_rand = torch.randn_like(xs_train_pre)
            xs_train_pre = 0.5 * xs_proj + 0.5 * xs_rand

    xs_test_post = xs_train_pre.clone()
    print("n_points = ", n_points)
    return xs_train_pre, xs_test_post
'''
def gen_coherence_tau(data_sampler, n_points, b_size, tau):
    """
    Génère xs_train et xs_test pour sparse recovery selon un degré de cohérence tau.
    
    Args:
        data_sampler : instance du sampler (CompressedSensingSampler)
        n_points : int, nombre d'exemples dans le prompt
        b_size : int, batch size
        tau : float entre 0 et 1, cohérence avec la base Phi
    Returns:
        xs_train_pre : Tensor (batch_size, n_points, d)
        xs_test_post : Tensor (batch_size, n_points, d)
    """
    xs_list = []

    for _ in range(b_size):
        xs, _, _, _ = data_sampler.sample(batch_size=1, n_points=n_points)
        xs = xs.squeeze(0)  # shape: (n_points, d)
        if tau == 0:
            xs = torch.randn_like(xs)
        else:
            Phi = torch.tensor(data_sampler.Phi, dtype=xs.dtype, device=xs.device)
            U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
            proj = U @ U.T
            if tau == 1:
                xs = xs @ proj
            elif 0 < tau < 1:
                xs = tau * (xs @ proj) + (1 - tau) * torch.randn_like(xs)
        xs_list.append(xs)

    xs_tensor = torch.stack(xs_list, dim=0)  # (batch_size, n_points, d)
    xs_train_pre = xs_tensor.clone()
    xs_test_post = xs_tensor.clone()
    return xs_train_pre, xs_test_post
'''
def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    conf=None,
    prompting_strategy=None,
):
    """
    Évalue un modèle sur la tâche Sparse Recovery en utilisant exclusivement
    gen_coherence_tau() pour générer les prompts.
    """

    assert num_eval_examples % batch_size == 0

    # --- Préparation des kwargs pour Sparse Recovery ---
    if conf is not None and hasattr(conf, "task") and hasattr(conf.task, "kwargs"):
        task_kwargs = dict(conf.task.kwargs)
    else:
        task_kwargs = {}

    # Fusion avec les overrides éventuels
    task_kwargs.update(data_sampler_kwargs)

    # Vérification des paramètres essentiels
    for arg in ["N", "d", "s", "tau"]:
        if arg not in task_kwargs:
            raise ValueError(f"⚠️ Paramètre '{arg}' manquant dans task_kwargs ou data_sampler_kwargs")

    # --- Data sampler ---
    data_sampler = get_data_sampler(
        data_name,
        N=task_kwargs["N"],
        d=task_kwargs["d"],
        s=task_kwargs["s"],
        Phi=task_kwargs.get("Phi", "identity"),
        tau=task_kwargs["tau"],
    )

    # --- Task sampler ---
    task_sampler_kwargs = dict(task_sampler_kwargs)
    task_sampler_kwargs.setdefault("N", task_kwargs["N"])
    task_sampler_kwargs.setdefault("d", task_kwargs["d"])
    task_sampler_kwargs.setdefault("s", task_kwargs["s"])
    task_sampler_kwargs.setdefault("Phi", task_kwargs.get("Phi", "identity"))
    task_sampler_kwargs.setdefault("tau", task_kwargs["tau"])

    task_sampler = get_task(task_name, **task_sampler_kwargs)

    # --- Boucle d'évaluation avec gen_coherence_tau() ---
    all_metrics = []
    tau = task_kwargs["tau"]

    for i in range(num_eval_examples // batch_size):
        xs, xs_p = gen_coherence_tau(data_sampler, n_points=n_points, b_size=batch_size, tau=tau)
        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)
    print("n_points = ", n_points)
    metrics = torch.cat(all_metrics, dim=0)
    return aggregate_metrics(metrics)
'''

def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    conf=None,
    tau=None,
    prompting_strategy=None,
):
    """
    Évalue un modèle sur la tâche Sparse Recovery avec perte cumulée par exemple in-context.
    Retourne les métriques agrégées via aggregate_metrics().
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    assert num_eval_examples % batch_size == 0

    # --- Préparation des kwargs pour Sparse Recovery ---
    if conf is not None and hasattr(conf, "task") and hasattr(conf.task, "kwargs"):
        task_kwargs = dict(conf.task.kwargs)
    else:
        task_kwargs = {}

    task_kwargs.update(data_sampler_kwargs)

    for arg in ["N", "d", "s"]:
        if arg not in task_kwargs:
            raise ValueError(f"⚠️ Paramètre '{arg}' manquant dans task_kwargs ou data_sampler_kwargs")

    # Si tau n'est pas fourni, on prend celui de la config ou 0
    if tau is None:
        tau = task_kwargs.get("tau", 0.0)

    # --- Data sampler ---
    data_sampler = get_data_sampler(
        data_name,
        N=task_kwargs["N"],
        d=task_kwargs["d"],
        s=task_kwargs["s"],
        Phi=task_kwargs.get("Phi", "identity"),
        tau=tau,
    )

    # --- Task sampler ---
    task_sampler_kwargs = dict(task_sampler_kwargs)
    task_sampler_kwargs.setdefault("N", task_kwargs["N"])
    task_sampler_kwargs.setdefault("d", task_kwargs["d"])
    task_sampler_kwargs.setdefault("s", task_kwargs["s"])
    task_sampler_kwargs.setdefault("Phi", task_kwargs.get("Phi", "identity"))
    task_sampler_kwargs.setdefault("tau", tau)

    task_sampler = get_task(task_name, **task_sampler_kwargs)

    # --- Boucle d'évaluation cumulative ---
    all_metrics = []

    for _ in range(num_eval_examples // batch_size):
        xs, _ = gen_coherence_tau(data_sampler, n_points=n_points, b_size=batch_size, tau=tau)
        b_size = xs.shape[0]
        metrics_batch = torch.zeros(b_size, n_points)

        for i in range(n_points):
            xs_prompt = xs[:, :i + 1, :]
            ys_pred = model(xs_prompt.to(xs_prompt.device), task_sampler.evaluate(xs_prompt).to(xs_prompt.device))
            ys_true = task_sampler.evaluate(xs_prompt)
            loss = task_sampler.get_metric()(ys_pred.cpu(), ys_true)
            if loss.ndim > 1:
                metrics_batch[:, i] = loss.mean(dim=1)
            else:
                metrics_batch[:, i] = loss

        all_metrics.append(metrics_batch)
    print("in eval_model")
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

    evaluation_kwargs = build_evals(conf)

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
    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute, conf=conf)
    return all_metrics

def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.points
    batch_size = conf.training.batch_size

    task_name = conf.task.name
    data_name = conf.task.name

    # Arguments de base (communs à toutes les configs)
    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    # Cas standard (par défaut)
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}

    # === Cas Sparse Recovery : on ajoute cohérence avec tau ===
    if task_name == "sparse_recovery":
        for tau in [0, 0.5, 1]:
            evaluation_kwargs[f"tau={tau}"] = {
                "prompting_strategy": f"coherence_tau_{tau}",
                "generator_fn": lambda data_sampler, n_pts, b_size, t=tau: gen_coherence_tau(
                    data_sampler, n_pts, b_size, tau=t
                ),
            }

        # Exemple d'évaluation bruitée pour SR
        evaluation_kwargs["noisy_sparse"] = {
            "task_sampler_kwargs": {"noise_std": 0.1},
        }
    
    # === Cas Matrix Factorization (exemple générique) ===
    elif task_name == "matrix_factorization":
        for rank in [2, 5, 10]:
            evaluation_kwargs[f"rank={rank}"] = {
                "task_sampler_kwargs": {"target_rank": rank}
            }
        evaluation_kwargs["noisyMF"] = {
            "task_sampler_kwargs": {"noise_std": 0.1},
        }

    # === Fusion finale avec base_kwargs ===
    for name, kwargs in evaluation_kwargs.items():
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs





def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False, conf=None):
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
        n_points = kwargs["n_points"]
        prompting_strategy = kwargs.get("prompting_strategy", "random")
        data_sampler_kwargs = kwargs.get("data_sampler_kwargs", {})

        # Fusionner avec conf.task.kwargs si dispo
        if conf is not None and hasattr(conf, "task") and hasattr(conf.task, "kwargs"):
            task_kwargs = dict(conf.task.kwargs)
            data_sampler_kwargs = {**task_kwargs, **data_sampler_kwargs}  # priorité à kwargs

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
                data_sampler_kwargs=data_sampler_kwargs,
                conf=conf,  # 🔑 on propage la config
            )

        all_metrics[eval_name] = metrics

    print("n_points = ", n_points)
    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics




def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
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
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
