import os
import json
from munch import Munch
import torch
import yaml
import pandas as pd
import numpy as np
import tqdm as tqdm

from sklearn.linear_model import Lasso
from sklearn.decomposition import TruncatedSVD

from models import build_model
from samplers import get_data_sampler
from tasks import get_task

# --- Chargement du modèle depuis un run ---
def get_model_from_run(run_path, step=-1, only_conf=False):
    import glob

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
        state_dict = checkpoint  # au cas où c'est juste le state_dict

    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    
    return model, conf


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size
    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}
    for tau in [0.0, 0.5, 1.0]:
        evaluation_kwargs[f"coherence-tau={tau}"] = {
            **base_kwargs,
            "data_sampler_kwargs": {"tau": tau}
        }

    return evaluation_kwargs


# --- Lecture des runs ---
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
                "task": conf.task.name,
                "run_id": run_id,
                "run_path": run_path,   # 🔹 ajouter run_path ici
            }
            all_data.append(run_dict)

    df = pd.DataFrame(all_data)
    return df


# --- Evaluation batch ---
def eval_batch(model, X, y):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = model(X.to(device), y.to(device)).detach()  # modèle prend X et y
    # Calcul d’une métrique simple, par exemple MSE
    loss = ((pred.cpu() - y) ** 2).mean(dim=1)
    return loss


def generate_eval_data(task_name, N, tau=0.5, **kwargs):
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
def eval_model(model, task_name, N, taus, Phi=None, d=None, s=None, n1=None, n2=None, rank=None):
    results = {}
    for tau in taus:
        if task_name == "sparse_recovery":
            X, y, w_star, a_star = generate_eval_data(
                task_name, N=N, tau=tau, d=d, s=s, Phi=Phi
            )
        elif task_name == "matrix_factorization":
            X, y, U_star, V_star = generate_eval_data(
                task_name, N=N, tau=tau, n1=n1, n2=n2, rank=rank
            )
        metrics = eval_batch(model, X, y)
        results[tau] = metrics
        print(f"Tau={tau} -> Metrics: {metrics}")
    return results


def get_run_metrics(run_path, task_name, N=16, Phi=None, d=None, s=None, n1=None, n2=None, rank=None):
    model, conf = get_model_from_run(run_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    taus = [0.0, 0.5, 1.0]
    metrics = eval_model(model, task_name, N, taus, Phi=Phi, d=d, s=s, n1=n1, n2=n2, rank=rank
    )
    return metrics

def compute_evals(evaluation_kwargs, conf):
    results = {}
    for eval_name, kwargs in evaluation_kwargs.items():
        print(f"🔹 Running evaluation: {eval_name}")

        task_name = kwargs["task_name"]
        n_dims = kwargs["n_dims"]
        n_points = kwargs["n_points"]
        data_sampler_kwargs = kwargs.get("data_sampler_kwargs", {})

        sampler = get_data_sampler(
            task_name,
            N=n_points,
            d=n_dims,
            **data_sampler_kwargs
        )

        results[eval_name] = {}

        # Baselines selon la tâche
        if task_name == "sparse_recovery":
            alphas = [1, 0.1, 0.01, 0.001, 0.0001]
            baselines = [Lasso(alpha=alpha, fit_intercept=False, max_iter=10000) for alpha in alphas]
        elif task_name == "matrix_factorization":
            n_components = min(n_dims, n_points)
            baselines = [TruncatedSVD(n_components=n_components) for _ in range(1)]  # un baseline SVD simple
            alphas = [None]  # pour itérer facilement
        else:
            baselines = []
            alphas = []

        for alpha, baseline in zip(alphas, baselines):
            name = f"Lasso_alpha={alpha}" if alpha is not None else "SVD_baseline"
            print(f"   ▶ Evaluating baseline: {name}")
            losses = []

            for _ in tqdm(range(conf.evaluation.n_batches)):
                X, y, w_star, a_star = sampler.sample()
                Phi = data_sampler_kwargs.get("Phi", None)
                if Phi is not None:
                    X_effectif = X @ Phi
                else:
                    X_effectif = X

                if task_name == "sparse_recovery":
                    baseline.fit(X_effectif.cpu().numpy(), y.cpu().numpy())
                    pred = torch.tensor(baseline.predict(X_effectif.cpu().numpy()), dtype=torch.float32)
                elif task_name == "matrix_factorization":
                    baseline.fit(X_effectif.cpu().numpy())
                    pred = torch.tensor(baseline.inverse_transform(baseline.transform(X_effectif.cpu().numpy())), dtype=torch.float32)
                else:
                    pred = torch.zeros_like(y)

                loss = ((pred - y) ** 2).mean(dim=1)
                losses.append(loss.mean().item())

            results[eval_name][name] = {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
            }

    return results

def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    return name


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
