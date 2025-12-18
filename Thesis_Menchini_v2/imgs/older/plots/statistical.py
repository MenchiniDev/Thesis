from rl_benchmark_plots import load_data, pretty_name


MEMORY_MODELS = [
    "PPO (16-frame memory)",
    "Recurrent PPO",
    "SAC (16-frame memory)"
]

NO_MEMORY_MODELS = [
    "PPO (no memory)",
    "SAC (no memory)"
]

def build_groups(models):
    mem_values_sr = []
    nomem_values_sr = []
    
    mem_values_prog = []
    nomem_values_prog = []

    for key, df in models.items():
        label = pretty_name(key)

        if label in MEMORY_MODELS:
            mem_values_sr.extend(df["success_rate"].values)
            mem_values_prog.extend(df["avg_ep_rew"].values)

        elif label in NO_MEMORY_MODELS:
            nomem_values_sr.extend(df["success_rate"].values)
            nomem_values_prog.extend(df["avg_ep_rew"].values)

    return mem_values_sr, nomem_values_sr, mem_values_prog, nomem_values_prog


import numpy as np

def cohens_d(a, b):
    na, nb = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*var_a + (nb-1)*var_b) / (na + nb - 2))
    return (mean_a - mean_b) / pooled

import matplotlib.pyplot as plt

def plot_group_boxplots(mem_sr, nomem_sr):
    data = [nomem_sr, mem_sr]
    labels = ["No Memory", "Memory"]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.ylabel("Success Rate")
    plt.title("Success Rate Distribution: Memory vs No Memory")
    plt.grid(axis="y")
    plt.tight_layout()


import seaborn as sns
import pandas as pd

def plot_violin(mem_sr, nomem_sr):
    df = pd.DataFrame({
        "success_rate": mem_sr + nomem_sr,
        "group": ["Memory"] * len(mem_sr) + ["No Memory"] * len(nomem_sr)
    })

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x="group", y="success_rate")
    plt.title("Success Rate Distribution")
    plt.tight_layout()

def plot_effect_size(d):
    plt.figure(figsize=(4,5))
    plt.bar(["Cohen d"], [d], color="steelblue")
    plt.axhline(0.2, color="gray", ls="--", label="small")
    plt.axhline(0.5, color="gray", ls="--", label="medium")
    plt.axhline(0.8, color="gray", ls="--", label="large")
    plt.ylabel("Effect Size (Cohen d)")
    plt.title("Effect Size of Memory vs No Memory")
    plt.legend()
    plt.tight_layout()

def plot_multivariate_centroids(mem_sr, nomem_sr, mem_prog, nomem_prog):
    import numpy as np

    mem_centroid = [np.mean(mem_sr), np.mean(mem_prog)]
    nomem_centroid = [np.mean(nomem_sr), np.mean(nomem_prog)]

    plt.figure(figsize=(6,5))
    plt.scatter(mem_centroid[0], mem_centroid[1], s=200, label="Memory", color="blue")
    plt.scatter(nomem_centroid[0], nomem_centroid[1], s=200, label="No Memory", color="red")

    plt.xlabel("Success Rate")
    plt.ylabel("Reward")
    plt.title("Multivariate Centroid Comparison (MANOVA)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_forest(means, cis, labels, metric_name):
    plt.figure(figsize=(6, 5))

    y = np.arange(len(means))

    plt.errorbar(means, y, xerr=cis, fmt='o', color="black", capsize=5)
    plt.yticks(y, labels)
    plt.xlabel(metric_name)
    plt.title(f"Means + 95% CI for {metric_name}")
    plt.grid(True, axis="x")
    plt.tight_layout()

import numpy as np

def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(values))
    return mean, ci95


def plot_progress_boxplots(mem_prog, nomem_prog):
    data = [nomem_prog, mem_prog]
    labels = ["No Memory", "Memory"]

    plt.figure(figsize=(6, 5))
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.ylabel("Avg Episode Reward")
    plt.title("Reward Distribution: Memory vs No Memory")
    plt.grid(axis="y")
    plt.tight_layout()



if __name__ == "__main__":
    models = load_data(base_path="./benchmark")
    
    from scipy.stats import mannwhitneyu

    mem_values_sr, nomem_values_sr, mem_values_prog, nomem_values_prog = build_groups(models)

    stat, p = mannwhitneyu(mem_values_sr, nomem_values_sr, alternative="greater")
    print("Success rate test: U =", stat, "p-value =", p)


    d = cohens_d(mem_values_sr, nomem_values_sr)
    print("Effect size (Cohen d):", d)

    from statsmodels.multivariate.manova import MANOVA
    import pandas as pd

    df_all = pd.DataFrame({
        "success": mem_values_sr + nomem_values_sr,
        "progress": mem_values_prog + nomem_values_prog,
        "group": ["mem"] * len(mem_values_sr) + ["nomem"] * len(nomem_values_sr)
    })

    ma = MANOVA.from_formula("success + progress ~ group", data=df_all)
    print(ma.mv_test())


    plot_group_boxplots(mem_values_sr, nomem_values_sr)
    plot_violin(mem_values_sr, nomem_values_sr)
    plot_effect_size(d)
    plot_multivariate_centroids(mem_values_sr, nomem_values_sr, mem_values_prog, nomem_values_prog)
    plot_progress_boxplots(mem_values_prog, nomem_values_prog)

    plt.show()