import numpy as np
from scipy.stats import beta

class LILY:
    def __init__(self, dim: int):
        self.dim: int = dim
        self.s: np.ndarray = np.zeros(dim)  # cumulative sum of clicked embeddings
        self.N: int = 0  # number of positive feedbacks (clicks)

    def update(self, x: np.ndarray) -> None:
        """
        Update model with a new positive feedback vector.

        Args:
            x (np.ndarray): A d-dimensional vector representing the clicked item.
        """
        self.s += x
        self.N += 1

    def score(self, x: np.ndarray) -> float:
        """
        Compute the average likelihood score for the given input vector.

        Args:
            x (np.ndarray): A d-dimensional vector to score.

        Returns:
            float: The average score based on per-dimension Beta likelihoods.
        """
        if self.N == 0:
            return 0.5  # default score before seeing any data

        alpha: np.ndarray = self.s + 1
        beta_param: np.ndarray = self.N - self.s + 1
        likelihoods: np.ndarray = beta.pdf(x, alpha, beta_param)
        return float(np.mean(likelihoods))

    def rerank(self, candidates: np.ndarray) -> np.ndarray:
        """
        Rerank candidates by their scores from high to low.

        Args:
            candidates (np.ndarray): shape (num_candidates, dim)

        Returns:
            np.ndarray: candidates sorted by score descending
        """
        scores = [self.score(x) for x in candidates]
        sorted_candidates = [x for _, x in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
        return np.array(sorted_candidates)
