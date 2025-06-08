# **LILY: Likelihood-based Interpretable Linear Yield-driven algorithm for Vectorized Contextual Bandits**

---

## Abstract

We introduce **LILY**, a novel contextual bandit algorithm designed for vectorized contexts under binary feedback. Unlike classical methods that rely on either hand-tuned confidence bounds (e.g., LinUCB) or non-gradual stochastic exploration (e.g., Thompson Sampling), LILY leverages per-dimension Beta distributions to model click likelihoods in a mathematically interpretable and adaptive fashion. It requires no parameter tuning, is highly resource-efficient, and remains stable even under sparse feedback. LILY assumes that the context is represented as a real-valued embedding vector with each coordinate bounded in $[0,1]$ and can be seamlessly applied to raw features, pretrained model outputs, or latent item representations. We emphasize its simplicity, interpretability, and practical flexibility, making it an appealing solution for modern real-time decision systems.

---

## Introduction

Contextual bandit algorithms have become a foundational component in many real-world applications such as online recommendation, personalized advertising, and adaptive decision-making systems. Despite the surge of neural-based models in this area, classical methods like **Thompson Sampling** \[1\] and **LinUCB** \[2\] remain highly valuable due to their low computational cost, interpretability, and stable performance under resource-constrained settings. However, each of these classic approaches has its own limitations: LinUCB requires careful tuning of confidence parameters to achieve balanced exploration, while Thompson Sampling often performs abrupt and non-gradual exploration, which may harm user experience in practice.

In this work, we propose **LILY (Likelihood-based Interpretable Linear Yield-driven algorithm)**, a novel contextual bandit algorithm that builds upon the intuition of per-dimension uncertainty modeling using **Beta distributions**. LILY assumes that the context is represented as a real-valued embedding vector with each coordinate bounded in $[0,1]$ and maintains a count-based belief update mechanism without relying on parameter tuning or explicit reward modeling. Compared to traditional approaches, LILY achieves adaptive and smooth exploration behavior, making it particularly suitable for scenarios with binary or sparse feedback and vectorized inputs such as pretrained embeddings or latent item factors. We highlight LILY’s simplicity, mathematical elegance, and practical utility as key advantages over both classical and deep-learning-based bandits.

---

## Related Work

### Thompson Sampling and Linear Bandits

**Thompson Sampling (TS)** is a well-established approach for exploration in multi-armed bandits based on Bayesian posterior sampling . In its classical form, TS assumes a Bernoulli reward model and maintains Beta distributions for each arm. While elegant and easy to implement, its inherent stochasticity can lead to unstable behavior, especially in early rounds. To address structured contexts, **LinUCB** and **LinTS** \[3\] extend bandit algorithms to the linear payoff setting, modeling the expected reward as a linear function of the context vector. Note that “LinTS” is a commonly used abbreviation in the literature for the algorithm introduced by Agrawal and Goyal (2013), though it was not explicitly named as such in the original paper. LinUCB uses confidence bounds for exploration, while LinTS performs posterior sampling over the model parameters.

### Embedding-Based Bandits

In recommendation systems, items and users are often represented as dense embedding vectors. Recent work explores how to incorporate such vectorized information into bandit frameworks. For example, **UBM-LinUCB** \[4\] introduces uncertainty-aware learning in the latent factor space, enhancing LinUCB with prior knowledge from collaborative filtering. Our approach shares a similar motivation leveraging embedding spaces but avoids reliance on confidence parameters by directly modeling click likelihoods using normalized Beta distributions. This results in a more adaptive and lightweight mechanism for vector-based contexts.

### Neural Bandits and Graph-based Methods

Neural contextual bandits have emerged to capture complex, nonlinear reward structures. One notable example is **Graph Neural Bandits (GNB)** \[5\], which integrates graph neural networks with contextual bandits for recommendation. While powerful, these methods often require high computational resources and are typically black-box in nature, making them harder to interpret and deploy in real-time systems. In contrast, LILY focuses on preserving interpretability and efficiency, providing an alternative pathway that sacrifices minimal performance while remaining mathematically grounded and suitable for sparse-feedback regimes.

---

## Algorithm Details

LILY is designed to balance interpretability, adaptivity, and low computational cost. It models user click feedback as independent observations on each embedding dimension and maintains Beta distributions to estimate the likelihood of clicks along each dimension. The algorithm computes a score for each candidate item by evaluating its embedding vector under the current per-dimension Beta distributions. This section introduces the key components of LILY’s design.

### Problem Setup

We consider a contextual bandit setting where at each time step $t$, the system is presented with a set of candidate items $\mathcal{I}_t = \{x_1, \dots, x_n\}$, where each item $x_i \in [0,1]^d$, with each coordinate in $[0,1]$. The system observes only the embedding $x_t$ of the item clicked by the user. The feedback is implicit and binary: only positive clicks are recorded; non-clicked items provide no feedback and are treated as missing.

Our goal is to maximize cumulative reward while maintaining a compact, interpretable, and parameter-free model.

### Why Beta Distributions?

Before describing the likelihood modeling, we motivate the choice of Beta distributions. We select Beta distributions because they have few parameters, are easy to control, and flexible enough to model shapes with modes anywhere within the interval $[0,1]$. This makes them particularly suitable for modeling binary feedback over continuous features bounded in this range. Their adaptive nature allows the distribution to sharpen as more data arrives, effectively capturing uncertainty and user preference dynamics. This simplicity and flexibility is also a key reason why the original Thompson Sampling algorithm employs Beta distributions as priors and posteriors in the classical multi-armed bandit setting.

### Likelihood Modeling with Beta Distributions

For each embedding dimension $j \in \{1, \dots, d\}$, LILY maintains two scalars:

* $s_j$: the cumulative sum of values in dimension $j$ from previously clicked item vectors
* $N$: the total number of observed clicks

At each step, we model the likelihood of a click on a new item $x = (x_1, \dots, x_d)$ using a Beta distribution for each dimension:

$$
\text{Beta}_j(x_j; \alpha_j, \beta_j) = \frac{x_j^{\alpha_j - 1} (1 - x_j)^{\beta_j - 1}}{B(\alpha_j, \beta_j)}
$$

where

$$
\alpha_j = s_j + 1
$$

$$
\beta_j = N - s_j + 1
$$

This choice ensures that each Beta distribution starts as a uniform prior $(\alpha = \beta = 1)$ and becomes sharper with more data. The score for item $x$ is computed as the average likelihood over all dimensions:

$$
\text{score}(x) = \frac{1}{d} \sum_{j=1}^d \text{BetaPDF}(x_j; \alpha_j, \beta_j)
$$

> $\text{BetaPDF}(x_j; \alpha_j, \beta_j)$ denotes the probability density function of the Beta distribution evaluated at $x_j$

The item with the highest score is selected for recommendation.

### Update Rule

Upon receiving a click, LILY updates the cumulative statistics as follows:

* For each dimension $j$, increment $s_j \leftarrow s_j + x_j$
* Increment $N \leftarrow N + 1$

LILY does not update on non-clicks, reflecting the positive-only feedback common in implicit recommendation systems. This design focuses learning on what users actively choose, enabling efficient preference modeling without needing explicit negatives. Irrelevant dimensions tend to exhibit high variance in their observed values, preventing the corresponding Beta distributions from forming sharp peaks—effectively downweighting their influence in scoring.

### Yield-Driven

Another notable advantage of LILY is that it is entirely deterministic—which is why we refer to it as Yield-Driven. Unlike exploration strategies that rely on random sampling (e.g., Thompson Sampling), LILY makes ranking decisions solely based on accumulated evidence and fixed probabilistic evaluations. This determinism enhances stability, debuggability, and reproducibility, making the algorithm particularly suitable for real-time systems where predictable behavior is essential.

---

## Experiments

To evaluate the effectiveness of the proposed **LILY** algorithm in learning user preferences through interaction, we conducted simulation experiments on two classic multi-class datasets: **Iris** and **Wine**, both available from `scikit-learn`. Each instance in these datasets belongs to a distinct class, which we interpret as a user preference.

### Experimental Setup

We simulate the user interaction process as follows:

* In each round, a pool of 20 candidate items is sampled uniformly at random from the dataset.
* The LILY model reranks the candidate items based on its current belief.
* The top-$k$ items (where $k$ is the number of unique classes in the dataset) are considered the system’s recommendations.
* If any of the top-$k$ items belong to the user’s preferred class, we record it as a "hit."
* The user then randomly selects one item from their preferred class (if available in the pool), and its feature vector is used as positive feedback to update the LILY model.
* This interaction loop is repeated for 20 rounds.
* For robustness, each experiment is repeated 3 times with different randomly selected preferred classes.

All features are normalized to the $[0, 1]$ range using MinMaxScaler. Candidate pools are shuffled in each round to prevent order bias.

### Results

The plots below show the **Top-k Hit Rate** (i.e., the proportion of preferred items among the top-$k$ results) over 20 rounds. Each line represents a different random preference run.

* **Iris dataset:**

![Iris Results](https://raw.githubusercontent.com/avengerandy/LILY/master/irisResult.png)

* **Wine dataset:**

![Wine Results](https://raw.githubusercontent.com/avengerandy/LILY/master/wineResult.png)

The results indicate that LILY is able to adapt quickly to user preferences, often achieving high hit rates within just a few rounds. This supports its applicability in cold-start or interactive recommendation scenarios, especially when embedding vectors are of high quality or the class separation is clear.

---

## Future Work

Several promising directions exist for extending LILY. One potential enhancement is to explore the use of Dirichlet distributions instead of independent Beta distributions, allowing the model to capture correlations between embedding dimensions. While this may improve accuracy, it introduces increased mathematical and computational complexity, potentially compromising the simplicity and interpretability of the method.

Beyond Dirichlet, a more flexible approach to modeling dependencies among embedding dimensions involves leveraging copula functions combined with Beta marginals. Copulas enable explicit modeling of complex, non-linear dependencies across dimensions without restricting the marginal distributions, thus potentially capturing richer interactions that Dirichlet or independent Betas may miss. However, this added expressiveness comes at the cost of more involved inference and potentially reduced interpretability.

Alternatively, covariance-based modeling offers a simpler way to capture linear dependencies among dimensions, suitable when the relationships are approximately Gaussian or linear. This approach can serve as a computationally cheaper intermediate step between fully independent Betas and complex copula models, balancing model complexity and performance.

Another direction involves partial and delayed feedback. In realistic environments, user clicks may be noisy, incomplete, or delayed. LILY currently treats all historical clicks equally; incorporating time decay or dynamic weighting could make the method more robust in evolving preference scenarios. However, we deliberately avoid complex dimension weighting at this stage to maintain LILY's simplicity and elegance.

---

## Conclusion

We introduced **LILY (Likelihood-based Interpretable Linear Yield-driven algorithm)**, a novel contextual bandit algorithm that leverages per-dimension Beta distributions over normalized embedding vectors to balance exploration and exploitation in a principled and interpretable way. LILY is parameter-free, low-resource, and adaptable to various input representations, including pre-trained embeddings and content-based vectors.

In contrast to classical methods such as LinUCB, LinTS, and deep neural bandits, LILY avoids costly model fitting or sampling while retaining competitive behavior through cumulative evidence accumulation. With its emphasis on simplicity, clarity, and practical applicability, LILY offers a compelling alternative for recommendation scenarios where transparency and efficiency are critical.

---

## References

\[1\] Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2017). A Tutorial on Thompson Sampling. arXiv preprint arXiv:1707.02038.

\[2\] Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. arXiv preprint arXiv:1003.0146.

\[3\] Agrawal, S., & Goyal, N. (2012). Thompson Sampling for Contextual Bandits with Linear Payoffs. arXiv preprint arXiv:1209.3352.

\[4\] He, X., An, B., Li, Y., Chen, H., Guo, Q., Li, X., & Wang, Z. (2020). Contextual User Browsing Bandits for Large-Scale Online Mobile Recommendation. arXiv preprint arXiv:2008.09368.

\[5\] Qi, Y., Ban, Y., & He, J. (2023). Graph Neural Bandits. arXiv preprint arXiv:2308.10808.
