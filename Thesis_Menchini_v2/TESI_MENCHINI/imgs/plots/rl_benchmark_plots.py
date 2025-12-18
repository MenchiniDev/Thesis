import pandas as pd
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
#        ORDINAMENTO MODELLI PER TUTTI I GRAFICI
# ---------------------------------------------------------

MODEL_ORDER_LABELS = [
    "PPO (no memory)",
    "PPO (16-frame memory)",
    "Recurrent PPO",
    "SAC (no memory)",
    "SAC (16-frame memory)",
    "tdmpc2",
]


def pretty_name(model_key: str) -> str:
    """
    Da nomi tipo:
        - 'PPO_final_v1'
        - 'PPO_nomem_final_v1'
        - 'SAC_nomem_final_v1'
        - 'recurrentPPO_final_v1'
    restituisce un nome leggibile.
    """
    name = model_key.lower()

    # Recurrent PPO
    if "recurrent" in name and "ppo" in name:
        return "Recurrent PPO"
    if "rppo" in name:
        return "Recurrent PPO"

    # PPO
    if "ppo" in name and "nomem" in name:
        return "PPO (no memory)"
    if "ppo" in name:
        return "PPO (16-frame memory)"

    # SAC
    if "sac" in name and "nomem" in name:
        return "SAC (16-frame memory)"  # come nel tuo mapping invertito
    if "sac" in name:
        return "SAC (no memory)"
    
    if "tdmpc2" in name:
        return "tdmpc2"

    return model_key


def ordered_model_keys(models: dict):
    """
    Restituisce le chiavi dei modelli ordinate secondo MODEL_ORDER_LABELS.
    """
    def sort_key(k: str):
        label = pretty_name(k)
        try:
            idx = MODEL_ORDER_LABELS.index(label)
        except ValueError:
            idx = len(MODEL_ORDER_LABELS)  # elementi non previsti in coda
        return (idx, label)

    return sorted(models.keys(), key=sort_key)


# ---------------------------------------------------------
#       FUNZIONI DI CARICAMENTO & SUPPORTO
# ---------------------------------------------------------

def add_steps_column(df: pd.DataFrame) -> pd.DataFrame:
    """Estrae gli steps dal nome del checkpoint e aggiunge 'steps'."""

    def extract_steps(path: str):
        base = os.path.basename(path)
        match = re.search(r"_([0-9]+)_steps", base)
        if match:
            return int(match.group(1))
        return None

    df = df.copy()
    df["steps"] = df["checkpoint"].apply(extract_steps)
    return df.sort_values("steps")


def load_data(base_path="./benchmark"):
    """
    Cerca ricorsivamente tutti i *_benchmark.csv sotto base_path.
    """
    models = {}
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.endswith("_benchmark.csv"):
                csv_path = os.path.join(root, fname)
                model_key = fname[:-len("_benchmark.csv")]
                df = pd.read_csv(csv_path)
                models[model_key] = df
                print(f"Caricato {csv_path} come '{model_key}' ({len(df)} run)")

    if not models:
        raise FileNotFoundError(f"Nessun *_benchmark.csv trovato sotto {base_path}")

    return models


# ---------------------------------------------------------
#                          GRAFICI
# ---------------------------------------------------------

def plot_progress_summary(models: dict):
    model_keys = ordered_model_keys(models)

    means, stds, labels = [], [], []
    for key in model_keys:
        df = models[key]
        vals = df["avg_ep_rew"].values
        means.append(vals.mean() * 100.0)
        stds.append(vals.std(ddof=1) * 100.0)
        labels.append(pretty_name(key))

    x = np.arange(len(model_keys))

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5, edgecolor="black")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("raw progress")
    plt.title("raw progress at best checkpoints (mean ± std)")
    plt.tight_layout()


def plot_success_rate_summary(models: dict):
    model_keys = ordered_model_keys(models)

    means, stds, labels = [], [], []
    for key in model_keys:
        df = models[key]
        vals = df["success_rate"].values
        means.append(vals.mean() * 100.0)
        stds.append(vals.std(ddof=1) * 100.0)
        labels.append(pretty_name(key))

    x = np.arange(len(model_keys))

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5, edgecolor="black")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 100)
    plt.title("Success rate at best checkpoints (mean ± std)")
    plt.tight_layout()


def plot_success_rate(models: dict):
    plt.figure()
    for key in ordered_model_keys(models):
        df = models[key]
        plt.plot(df["steps"], df["success_rate"], marker="o", label=pretty_name(key))
    plt.xlabel("Training steps")
    plt.ylabel("Success rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_avg_reward(models: dict):
    plt.figure()
    for key in ordered_model_keys(models):
        df = models[key]
        plt.plot(df["steps"], df["avg_ep_rew"], marker="o", label=pretty_name(key))
    plt.xlabel("Training steps")
    plt.ylabel("Average episode reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_avg_ep_len(models: dict):
    plt.figure()
    for key in ordered_model_keys(models):
        df = models[key]
        plt.plot(df["steps"], df["avg_ep_len"], marker="o", label=pretty_name(key))
    plt.xlabel("Training steps")
    plt.ylabel("Episode length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_failure_breakdown(models: dict):
    failure_cols = ["oob", "collide_rock", "danger_slope", "rollover", "timeout"]

    for key in ordered_model_keys(models):
        df = models[key]
        steps = df["steps"].astype(str)
        bottom = None

        plt.figure()
        for col in failure_cols:
            if col not in df.columns:
                continue
            vals = df[col]
            if bottom is None:
                plt.bar(steps, vals, label=col)
                bottom = vals
            else:
                plt.bar(steps, vals, bottom=bottom, label=col)
                bottom = bottom + vals

        plt.xlabel("Training steps")
        plt.ylabel("Episodes")
        plt.title(f"Failure breakdown ({pretty_name(key)})")
        plt.legend()
        plt.tight_layout()


def plot_reward_vs_ep_len(models: dict):
    plt.figure()
    for key in ordered_model_keys(models):
        df = models[key]
        plt.scatter(df["avg_ep_len"], df["avg_ep_rew"], label=pretty_name(key))
    plt.xlabel("Avg episode length")
    plt.ylabel("Avg reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_best_checkpoint_events(models: dict):
    failure_cols = ["oob", "collide_rock", "danger_slope",
                    "rollover", "timeout", "success"]

    model_keys = ordered_model_keys(models)
    n_models = len(model_keys)

    x = np.arange(len(failure_cols))
    width = 0.8 / max(1, n_models)

    plt.figure(figsize=(12, 6))

    for i, key in enumerate(model_keys):
        df = models[key]
        means = [df[col].mean() for col in failure_cols]
        stds = [df[col].std(ddof=1) for col in failure_cols]

        offset = (i - (n_models - 1) / 2) * width
        x_pos = x + offset

        plt.bar(
            x_pos, means, width=width,
            yerr=stds, capsize=4, edgecolor="black",
            label=pretty_name(key)
        )

    plt.xticks(x, failure_cols)
    plt.ylim(0, 100)
    plt.ylabel("Episodes (mean over runs)")
    plt.title("Event types at best checkpoints (mean ± std)")
    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------
#                           MAIN
# ---------------------------------------------------------

def main():
    models = load_data(base_path="./benchmark")

    plot_progress_summary(models)
    plot_success_rate_summary(models)
    plot_best_checkpoint_events(models)

    # Opzionali:
    # plot_success_rate(models)
    # plot_avg_reward(models)
    # plot_avg_ep_len(models)
    # plot_failure_breakdown(models)
    # plot_reward_vs_ep_len(models)

    plt.show()


if __name__ == "__main__":
    main()
