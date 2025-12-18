import re
import numpy as np
import matplotlib.pyplot as plt

# === INCOLLA QUI I TUOI LOG ===
log_text = """
[TIMING] total= 470.6 ms | depth=   1.1 | slopes= 444.1 | policy=  25.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 618.8 ms | depth=   1.1 | slopes= 594.2 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 472.5 ms | depth=   1.1 | slopes= 447.9 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 529.2 ms | depth=   1.1 | slopes= 504.2 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total= 477.3 ms | depth=   1.1 | slopes= 453.6 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 419.7 ms | depth=   1.1 | slopes= 395.0 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 406.1 ms | depth=   1.3 | slopes= 381.8 | policy=  23.0
[INFO] HFOV = 0.980 rad
[TIMING] total= 468.4 ms | depth=   1.4 | slopes= 443.3 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 405.8 ms | depth=   1.3 | slopes= 380.8 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 704.5 ms | depth=   1.4 | slopes= 679.7 | policy=  23.4
[INFO] HFOV = 0.980 rad
[TIMING] total= 462.9 ms | depth=   1.1 | slopes= 435.7 | policy=  26.1
[INFO] HFOV = 0.980 rad
[INFO] HFOV = 0.980 rad
[TIMING] total= 537.6 ms | depth=   1.3 | slopes= 513.7 | policy=  22.7
[TIMING] total= 547.1 ms | depth=   1.4 | slopes= 522.1 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 715.4 ms | depth=   1.4 | slopes= 691.6 | policy=  22.4
[INFO] HFOV = 0.980 rad
[TIMING] total= 518.5 ms | depth=   1.1 | slopes= 493.7 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 359.2 ms | depth=   1.1 | slopes= 334.4 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 332.6 ms | depth=   1.4 | slopes= 307.6 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 533.4 ms | depth=   1.4 | slopes= 508.0 | policy=  24.1
[INFO] HFOV = 0.980 rad
[TIMING] total= 471.7 ms | depth=   1.4 | slopes= 446.8 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 363.1 ms | depth=   1.4 | slopes= 336.7 | policy=  25.0
[INFO] HFOV = 0.980 rad
[TIMING] total= 304.3 ms | depth=   1.4 | slopes= 279.1 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 625.4 ms | depth=   1.4 | slopes= 590.9 | policy=  33.1
[INFO] HFOV = 0.980 rad
[TIMING] total= 406.3 ms | depth=   1.4 | slopes= 378.0 | policy=  26.9
[INFO] HFOV = 0.980 rad
[TIMING] total= 356.7 ms | depth=   1.3 | slopes= 331.4 | policy=  24.1
[INFO] HFOV = 0.980 rad
[TIMING] total= 269.3 ms | depth=   1.5 | slopes= 245.3 | policy=  22.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 350.7 ms | depth=   1.4 | slopes= 325.5 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 346.6 ms | depth=   1.4 | slopes= 321.4 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 418.2 ms | depth=   1.4 | slopes= 393.3 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 237.7 ms | depth=   1.4 | slopes= 213.6 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 264.3 ms | depth=   1.4 | slopes= 240.2 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 526.1 ms | depth=   1.4 | slopes= 501.1 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 346.2 ms | depth=   1.4 | slopes= 321.0 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 222.7 ms | depth=   1.4 | slopes= 197.7 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 699.4 ms | depth=   1.8 | slopes= 673.7 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total= 423.2 ms | depth=   1.1 | slopes= 398.5 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 363.2 ms | depth=   1.4 | slopes= 337.2 | policy=  24.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 283.2 ms | depth=   1.4 | slopes= 259.1 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 273.9 ms | depth=   1.4 | slopes= 250.1 | policy=  22.3
[INFO] HFOV = 0.980 rad
[TIMING] total= 553.8 ms | depth=   1.3 | slopes= 528.9 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 307.6 ms | depth=   1.4 | slopes= 282.6 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 297.9 ms | depth=   1.4 | slopes= 272.7 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 323.1 ms | depth=   1.4 | slopes= 298.3 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 363.1 ms | depth=   1.4 | slopes= 337.9 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 251.2 ms | depth=   1.4 | slopes= 225.5 | policy=  24.3
[INFO] HFOV = 0.980 rad
[TIMING] total= 104.7 ms | depth=   1.5 | slopes=  77.8 | policy=  25.3
[INFO] HFOV = 0.980 rad
[TIMING] total= 293.5 ms | depth=   1.4 | slopes= 267.9 | policy=  24.2
[INFO] HFOV = 0.980 rad
[TIMING] total= 272.7 ms | depth=   1.4 | slopes= 247.7 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 195.4 ms | depth=   1.4 | slopes= 170.3 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 384.8 ms | depth=   1.4 | slopes= 359.2 | policy=  24.2
[INFO] HFOV = 0.980 rad
[TIMING] total= 283.0 ms | depth=   1.4 | slopes= 257.9 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 109.1 ms | depth=   1.4 | slopes=  85.0 | policy=  22.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 127.2 ms | depth=   1.4 | slopes= 101.8 | policy=  24.1
[INFO] HFOV = 0.980 rad
[TIMING] total= 334.2 ms | depth=   1.4 | slopes= 310.3 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 277.8 ms | depth=   1.4 | slopes= 253.6 | policy=  22.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 153.8 ms | depth=   1.4 | slopes= 128.6 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total= 225.7 ms | depth=   1.6 | slopes= 199.0 | policy=  25.0
[INFO] HFOV = 0.980 rad
[TIMING] total= 110.9 ms | depth=   1.3 | slopes=  86.1 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 187.0 ms | depth=   1.4 | slopes= 161.8 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 177.2 ms | depth=   1.5 | slopes= 152.0 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 175.5 ms | depth=   1.4 | slopes= 151.4 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 193.3 ms | depth=   1.4 | slopes= 169.2 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 116.8 ms | depth=   1.4 | slopes=  91.6 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 119.7 ms | depth=   1.4 | slopes=  95.8 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 144.5 ms | depth=   1.4 | slopes= 120.2 | policy=  22.9
[INFO] HFOV = 0.980 rad
[TIMING] total=  95.6 ms | depth=   1.4 | slopes=  71.2 | policy=  23.1
[INFO] HFOV = 0.980 rad
[TIMING] total= 138.9 ms | depth=   1.4 | slopes= 113.8 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 168.2 ms | depth=   1.7 | slopes= 141.3 | policy=  25.2
[INFO] HFOV = 0.980 rad
[TIMING] total=  79.9 ms | depth=   1.4 | slopes=  54.9 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  71.5 ms | depth=   1.4 | slopes=  46.5 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 104.4 ms | depth=   1.4 | slopes=  79.4 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total=  78.6 ms | depth=   1.4 | slopes=  53.7 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 284.3 ms | depth=   1.4 | slopes= 260.3 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  58.2 ms | depth=   1.4 | slopes=  34.5 | policy=  22.3
[INFO] HFOV = 0.980 rad
[TIMING] total= 110.4 ms | depth=   1.4 | slopes=  85.4 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  83.3 ms | depth=   1.4 | slopes=  58.1 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 127.0 ms | depth=   1.4 | slopes= 101.9 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total=  59.2 ms | depth=   1.4 | slopes=  34.7 | policy=  23.0
[INFO] HFOV = 0.980 rad
[TIMING] total=  76.9 ms | depth=   1.2 | slopes=  51.3 | policy=  24.4
[INFO] HFOV = 0.980 rad
[TIMING] total=  70.1 ms | depth=   1.4 | slopes=  45.0 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total=  58.4 ms | depth=   1.4 | slopes=  34.4 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 122.2 ms | depth=   1.4 | slopes=  96.9 | policy=  24.0
[INFO] HFOV = 0.980 rad
[TIMING] total= 231.1 ms | depth=   1.4 | slopes= 206.0 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 220.4 ms | depth=   1.4 | slopes= 195.2 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total=  62.4 ms | depth=   1.4 | slopes=  35.4 | policy=  25.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 151.7 ms | depth=   1.4 | slopes= 126.5 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 133.3 ms | depth=   1.4 | slopes= 108.1 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total=  88.1 ms | depth=   1.4 | slopes=  63.0 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 109.2 ms | depth=   1.4 | slopes=  83.8 | policy=  24.1
[INFO] HFOV = 0.980 rad
[TIMING] total=  83.2 ms | depth=   1.4 | slopes=  54.4 | policy=  27.3
[INFO] HFOV = 0.980 rad
[TIMING] total=  59.6 ms | depth=   1.4 | slopes=  35.7 | policy=  22.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 121.5 ms | depth=   1.4 | slopes=  95.5 | policy=  24.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  57.6 ms | depth=   1.4 | slopes=  33.5 | policy=  22.8
[INFO] HFOV = 0.980 rad
[TIMING] total=  57.1 ms | depth=   1.4 | slopes=  33.2 | policy=  22.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 144.2 ms | depth=   1.4 | slopes= 119.1 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 152.6 ms | depth=   1.4 | slopes= 127.3 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total= 151.1 ms | depth=   1.4 | slopes= 125.9 | policy=  23.8
[INFO] HFOV = 0.980 rad
[TIMING] total= 129.0 ms | depth=   1.4 | slopes= 104.1 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total=  68.9 ms | depth=   1.4 | slopes=  44.8 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 169.0 ms | depth=   1.4 | slopes= 144.0 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 212.1 ms | depth=   1.6 | slopes= 184.5 | policy=  26.0
[INFO] HFOV = 0.980 rad
[TIMING] total=  68.3 ms | depth=   1.4 | slopes=  44.4 | policy=  22.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 136.5 ms | depth=   1.4 | slopes= 111.5 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  68.0 ms | depth=   1.4 | slopes=  42.7 | policy=  24.0
[INFO] HFOV = 0.980 rad
[TIMING] total=  70.3 ms | depth=   1.4 | slopes=  42.4 | policy=  26.6
[INFO] HFOV = 0.980 rad
[TIMING] total= 267.4 ms | depth=   1.3 | slopes= 242.6 | policy=  23.4
[INFO] HFOV = 0.980 rad
[TIMING] total= 139.7 ms | depth=   1.4 | slopes= 114.7 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  81.2 ms | depth=   1.4 | slopes=  56.3 | policy=  23.5
[INFO] HFOV = 0.980 rad
[TIMING] total=  61.3 ms | depth=   1.4 | slopes=  37.3 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 131.4 ms | depth=   1.3 | slopes= 106.2 | policy=  23.9
[INFO] HFOV = 0.980 rad
[TIMING] total=  79.8 ms | depth=   1.4 | slopes=  54.7 | policy=  23.7
[INFO] HFOV = 0.980 rad
[TIMING] total=  94.6 ms | depth=   1.9 | slopes=  67.4 | policy=  25.3
[INFO] HFOV = 0.980 rad
[TIMING] total=  60.8 ms | depth=   1.5 | slopes=  36.7 | policy=  22.7
[INFO] HFOV = 0.980 rad
[TIMING] total= 131.3 ms | depth=   1.4 | slopes= 106.4 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  90.4 ms | depth=   1.3 | slopes=  65.5 | policy=  23.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  93.8 ms | depth=   1.4 | slopes=  69.8 | policy=  22.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 245.2 ms | depth=   1.4 | slopes= 219.1 | policy=  24.6
[INFO] HFOV = 0.980 rad
[TIMING] total=  59.2 ms | depth=   1.4 | slopes=  35.3 | policy=  22.5
[INFO] HFOV = 0.980 rad
[TIMING] total= 110.3 ms | depth=   1.4 | slopes=  85.1 | policy=  23.8
"""

# ===== Parse log =====
pattern = r"total=\s*([\d.]+)\s*ms\s*\|\s*depth=\s*([\d.]+)\s*\|\s*slopes=\s*([\d.]+)\s*\|\s*policy=\s*([\d.]+)"
matches = re.findall(pattern, log_text)

totals = np.array([float(m[0]) for m in matches])
depth = np.array([float(m[1]) for m in matches])
slopes = np.array([float(m[2]) for m in matches])
policy = np.array([float(m[3]) for m in matches])

means = np.array([totals.mean(), depth.mean(), slopes.mean(), policy.mean()])
stds  = np.array([totals.std(ddof=1), depth.std(ddof=1), slopes.std(ddof=1), policy.std(ddof=1)])

print("Means  [total, depth, slopes, policy]:", means)
print("Stddev [total, depth, slopes, policy]:", stds)

# ===== Plot =====
labels = ["total", "depth", "slopes", "policy"]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(7, 3))

bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.9)

# colore arancione tipo figura precedente
for b in bars:
    b.set_color("orange")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Time per iteration [ms]")

# Titolo in due righe, stile figura
ax.set_title("Raspberry Pi 4 – Deterministic slope pipeline\nPer-phase mean and standard deviation")

# Linea rossa a 250 ms con etichetta
limit = 250
ax.axhline(limit, linestyle="--", color="red", linewidth=1)
ax.text(
    x[-1] + 0.15,  # un po' a destra dell'ultimo bar
    limit + 3,
    "RPi 5 time budget = 250 ms",
    color="red",
    fontsize=8,
    va="bottom"
)

ax.set_ylim(0, max(means + stds)*1.3)
ax.grid(axis="y", linestyle=":", alpha=0.4)

fig.tight_layout()
plt.show()
