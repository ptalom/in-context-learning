import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "sparse_recovery": [
        "Transformer",
        "Lasso (alpha=0.01)",
        "Least Squares",
    ],
    "matrix_factorization": [
        "Transformer",
        "Lasso (alpha=0.01)",
        "Nuclear Norm",
    ],
}



def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    # choisir les modèles si on en passe en argument
    if models is not None:
        metrics = {k: metrics[k] for k in models if k in metrics}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")

    for name, vs in metrics.items():
        mean = vs["mean"]
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]

        ax.plot(mean, "-", label=name, color=palette[color % 10], lw=2)
        ax.fill_between(range(len(low)), low, high, alpha=0.3, color=palette[color % 10])
        color += 1

    ax.set_xlabel("in-context examples")
    ax.set_ylabel("normalized error")
    if metrics:
        ax.set_xlim(-1, len(next(iter(metrics.values()))["mean"]) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(6, 4)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax

# Exemple d'utilisation


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        # Vérifie si la ligne est valide
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r["task"], r["run_id"])
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r["model"], r["run_id"])
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name:
                    model_name = r["model"]
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)

                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r["task"] in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r["task"] == "sparse_linear_regression":
                    normalization = int(r["kwargs"].split("=")[-1])
                if r["task"] == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v

                processed_results[model_name] = m_processed

            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)

            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}

            all_metrics[eval_name].update(processed_results)

    return all_metrics
