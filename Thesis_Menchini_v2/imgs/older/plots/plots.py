import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
# df = pd.read_csv('./data/PPO_final_v1_train_steps.csv')
df = pd.read_csv('./data/rPPO_final_v1_train_steps.csv')
#df = pd.read_csv('./data/PPO_nomem_final_v1_train_steps.csv')
#df = pd.read_csv('./data/SAC_nomem_final_v1_train_steps.csv')
#df = pd.read_csv('./data/SAC_final_v1_train_steps.csv')
## columns ##
#---------------------------------------------------
# timesteps,
# mean_step_reward,
# mean_penalty_total,
# mean_penalty_rocks,
# mean_penalty_slopes,
# mean_progress,
# mean_goal_reward,
# mean_oob_reward,
# mean_total_reward,
# mean_episode_return,
# mean_episode_length, 
# mean_dist_to_goal_m,
# safe_fraction
#---------------------------------------------------

#togli tutte le righe dopo la 500
print(df.shape)
df.drop((i for i in range(len(df)) if i % 2 == 0), inplace=True)
print(df.shape)

# Helper: scatter + linear fit
def scatter_with_fit(x, y, label):
    plt.scatter(x, y, s=25, marker="x", linewidths=2)  # X invece di cerchi
    coeffs = np.polyfit(x, y, 3)
    fit_line = np.polyval(coeffs, x)
    plt.plot(x, fit_line, label=f"{label} (fit)")
    return coeffs

# ---------------------------------------------------
# FIGURE 1: Reward Shaping Metrics
# ---------------------------------------------------
plt.figure(figsize=(10,6))

for col in [
    "mean_progress",
    "mean_goal_reward",
    "mean_total_reward"
]:
    scatter_with_fit(df["timesteps"], df[col], col)

plt.xlabel("Timesteps")
plt.ylabel("Reward Components")
plt.title("Reward Shaping Metrics With Scatter + Poly Fit")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# FIGURE 1.5: Episode Metrics
# ---------------------------------------------------

plt.figure(figsize=(10,6))

for col in [
    "mean_episode_return",
    "mean_episode_length"
]:
    scatter_with_fit(df["timesteps"], df[col], col)

plt.xlabel("Timesteps")
plt.ylabel("Reward Components")
plt.title("Episode metrics With Scatter + Poly Fit")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# FIGURE 2: Goal Metrics
# ---------------------------------------------------
plt.figure(figsize=(10,6))

for col in [
    "mean_dist_to_goal_m",
    "mean_progress",
    "mean_goal_reward"
]:
    scatter_with_fit(df["timesteps"], df[col], col)

plt.xlabel("Timesteps")
plt.ylabel("Goal Metrics")
plt.title("Goal Metrics With Scatter + Poly Fit")
plt.legend()
plt.tight_layout()
plt.show()
