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
    """
    Trace l'évolution des erreurs pour différents modèles.
    """
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models if k in metrics}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1

    ax.set_xlabel("in-context examples")
    ax.set_ylabel("normalized error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


import pandas as pd

def collect_results(df, task_name=None, valid_row=None):
    """
    Transforme un DataFrame de résultats en un format exploitable pour le plotting.
    
    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les résultats des runs (doit avoir au moins ['task', 'tau', 'n_points', 'metric'])
    task_name : str, optionnel
        Filtrer sur une tâche spécifique.
    valid_row : function, optionnel
        Fonction qui prend une ligne et retourne True si elle doit être conservée.
    
    Retour
    ------
    pd.DataFrame
        DataFrame filtré et prêt pour le plotting.
    """
    if task_name is not None:
        df = df[df['task'] == task_name]
    
    if valid_row is not None:
        df = df[df.apply(valid_row, axis=1)]
    
    # Reset index pour éviter les problèmes
    df = df.reset_index(drop=True)
    
    # Vérifier les colonnes attendues
    expected_cols = ['task', 'tau', 'n_points', 'metric']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"La colonne attendue '{col}' est absente du DataFrame")
    
    return df


