from lily import LILY
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def run_simulation(X, y, rounds: int = 20, pool_size: int = 20) -> list[float]:
    # Simulate a user's preferred class
    X, y = shuffle(X, y)
    preferred = np.random.choice(np.unique(y))
    lily = LILY(dim=X.shape[1])

    top_k_hit_rates = []

    for _ in range(rounds):
        # Sample a pool of candidates
        indices = np.random.choice(len(X), size=pool_size, replace=False)
        candidates = X[indices]
        labels = y[indices]

        # Rerank all candidates using LILY
        reranked = lily.rerank(candidates)
        top_k = reranked[:len(np.unique(y))]

        # Get the indices of top_k in candidates
        top_k_indices = [
            np.where(np.all(candidates == x, axis=1))[0][0] for x in top_k
        ]
        top_k_labels = labels[top_k_indices]

        # Compute hit rate (i.e., proportion of preferred items in top-k)
        hit = np.sum(top_k_labels == preferred) / len(top_k_labels)
        top_k_hit_rates.append(hit)

        # Simulate user clicking one of their preferred items (if any in pool)
        preferred_indices = [i for i, label in enumerate(labels) if label == preferred]
        if preferred_indices:
            i = np.random.choice(preferred_indices)
            lily.update(candidates[i])

    return top_k_hit_rates

# Run the iris experiment 3 times, each with a different random preference
plt.figure(figsize=(10, 6))

data = load_iris()
X = MinMaxScaler().fit_transform(data.data)
y = data.target
for run in range(3):
    hit_rates = run_simulation(X, y)
    plt.plot(hit_rates, label=f"Run {run + 1}")

plt.xlabel("Rounds")
plt.ylabel("Top-k Hit Rate")
plt.title("LILY Bandit Performance on Iris Dataset (3 Random Preferences)")
plt.legend()
plt.savefig("irisResult.png")

# Run the wine experiment 3 times, each with a different random preference
plt.figure(figsize=(10, 6))

data = load_wine()
X = MinMaxScaler().fit_transform(data.data)
y = data.target
for run in range(3):
    hit_rates = run_simulation(X, y)
    plt.plot(hit_rates, label=f"Run {run + 1}")

plt.xlabel("Rounds")
plt.ylabel("Top-k Hit Rate")
plt.title("LILY Bandit Performance on Wine Dataset (3 Random Preferences)")
plt.legend()
plt.savefig("wineResult.png")
