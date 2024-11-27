The attempt is based on Prof.Yuhua Zhu's paper [A consensus-based global optimization method for high dimensional machine learning problems](https://www.esaim-cocv.org/articles/cocv/pdf/2021/01/cocv190163.pdf)
# 1.Algorithm Design
### Initialization
* Generate Initial Particles: Start by generating a set of particles $`\left\{X_0^j \in \mathbb{R}^d\right\}_{j=1}^N$ from - a distribution $\rho_0`$.
* Initialize Remainder Set: Set the remainder set $`\mathcal{R}_0`$ to be empty.
### Iterative Process
For each iteration $`k=0,1,2, \ldots`$, perform the following steps:
#### Step 1: Batch Formation
 * Concatenate and Permute: Form a list $\mathcal{I}_k$ by concatenating the remainder set $`\mathcal{R}_k`$ with a random permutation $\mathcal{P}_k$ of indices $`\{1,2, \ldots, N\}`$.

 * Batch Division: Divide $`\mathcal{I}_k`$ into $`q=\left\lfloor\frac{N+\left|\mathcal{R}_k\right|}{M}\right\rfloor`$ batches of size $M$, resulting in batches $`B_1^k, B_2^k, \ldots, B_q^k`$. The remaining indices form the new remainder set $`\mathcal{R}_{k+1}`$.
#### Step 2: Batch Processing
For each batch $`B_\theta^k`$, perform the following:
1. Function Evaluation:
- Calculate or approximate the function values $`L^j=L\left(X^j\right)`$ for all $`j \in B_\theta^k`$.
- If $`L(x)`$ is complex, use a mini-batch approach to approximate it: $`\hat{L}^j=\frac{1}{m} \sum_{i \in A_\theta^k} \ell_i\left(X^j\right)`$ where $`A_\theta^k`$ is a random subset of indices.
2. Weighted Average Update:

Compute the weighted average $`\bar{x}_{k, \theta}^*: \bar{x}_{k, \theta}^*=\frac{1}{\sum_{j \in B_\theta^k} \mu_j} \sum_{j \in B_\theta^k} X^j \mu_j`$ where $`\mu_j=e^{-\beta L^j}$ or $e^{-\beta \hat{L}^j}`$.

3. Particle Update:
- Update each particle $`X^j`$ for $`j \in \mathcal{J}_{k, \theta}`$ :

```math
X^j \leftarrow X^j-\lambda \gamma_{k, \theta}\left(X^j-\bar{x}_{k, \theta}^*\right)+\sigma_{k, \theta} \sqrt{\gamma_{k, \theta}} \sum_{i=1}^d \vec{e}_i\left(X^j-\bar{x}_{k, \theta}^*\right)_i z_i^j
```
where $`z_i^j \sim \mathcal{N}(0,1)$ and $\gamma_{k, \theta}`$ is the learning rate.

- Update Options:
Partial Updates: $`\mathcal{J}_{k, \theta}=B_\theta^k`$
Full Updates: $`\mathcal{J}_{k, \theta}=\{1, \ldots, N\}`$
#### Step 3: Stopping Criterion
Check if the stopping criterion is met: $`\frac{1}{d}\left\|\Delta \bar{x}^*\right\|_2^2 \leq \epsilon`$ where $`\|\cdot\|_2`$ is the Euclidean norm and $`\Delta \bar{x}^*`$ is the change in the weighted average between iterations.

If the stopping criterion is not satisfied, repeat Steps 1 and 2.
# 2. Key observations from the comparison:

### Convergence Rate
The argmin method shows faster convergence to the optimal solution, as evidenced by the steeper descent in the convergence plot.

### Final Solution Quality

The argmin method achieved a significantly better final solution  compared to the weighted average method. The argmin method's mean position is closer to the true optimum (0,0) for our test function
### Stability 
The convergence plot shows that the argmin method has a more stable descent path with fewer fluctuations compared to the weighted average method.

The superior performance of the argmin method in this case can be attributed to its direct selection of the best-performing particle in each batch, which provides a stronger optimization signal compared to the weighted average approach.

# 3.Theoretical properties

### Method 1: Weighted Average Update

```math
\bar{x}_{k, \theta}^*=\frac{1}{\sum_{j \in B_\theta^k} \mu_j} \sum_{j \in B_\theta^k} X^j \mu_j, \quad with \quad \mu_j=e^{-\beta L^j}
```



Theoretical Properties:

1. Consensus Property:
- Under this update rule, as $\beta \to \infty$, the weighted average converges to:
```math
  \lim_{\beta \to \infty} \bar{x}_{k, \theta}^* = \underset{X^j \in B_\theta^k}{argmin} L(X^j)
```

2. Gradient Flow:
- The continuous-time limit of this method yields the gradient flow:
```math
  \frac{d}{dt}X_t = -\nabla V(X_t)
```

where 
```math
V(x) = -\frac{1}{\beta}\log\left(E[e^{-\beta L(X)}]\right)
```

3. Free Energy Dissipation:

- The method minimizes the free energy functional:
$$F(X) = \mathbb{E}[L(X)] + \frac{1}{\beta}E[\log(\rho(X))]$$
where $$\rho(X)$$ is the probability density of particles.

### Method 2: Argmin Update

```math
\bar{x}_k^*=\underset{X^j \in B_\theta^k}{argmin} L(X^j)
```

Theoretical Properties:

1. Deterministic Selection:
- This update directly selects the best particle, leading to a deterministic evolution without noise.

2. Convergence Rate:
- Under suitable conditions (L-smoothness and $$\mu$$-strong convexity), the convergence rate is:

```math
  E[L(\bar{x}_k^*) - L(x^*)] \leq (1-\alpha)^k[L(\bar{x}_0^*) - L(x^*)]
```

  where $\alpha$  depends on batch size $$M$$ and condition number $$\kappa = L/\mu$$.

3. Stability Analysis:
- The stability region is characterized by:
```math
\|\bar{x}_{k+1}^* - x^*\| \leq (1-\lambda\gamma)\|\bar{x}_k^* - x^*\| + \sigma\sqrt{\gamma}\|\xi_k\|
```
where $`x^{*}`$ is the global minimizer.

### Comparison of Theoretical Guarantees:

1. Exploration vs Exploitation:
- Weighted Average: Better exploration due to contribution from all particles
- Argmin: Stronger exploitation of good solutions

2. Convergence Properties:
- Weighted Average:
```math
E[\|\bar{x}_k^* - x^*\|^2] \leq C_1e^{-\alpha t} + \frac{C_2}{\beta}
```

- Argmin:
```math
E[\|\bar{x}_k^* - x^*\|^2] \leq C_3e^{-\lambda t}
```
where $`C_1, C_2, C_3`$ are constants.

3. Noise Sensitivity:
- The weighted average method has natural noise resistance due to averaging:
```math
Var(\bar{x}_{k,\theta}^*) \propto \frac{1}{|B_\theta^k|}
```

- The argmin method's variance depends on the noise in the best particle:
$$Var(\bar{x}_k^*) \propto \sigma^2$$

4. Global vs Local Convergence:
- Weighted Average: Better global convergence properties due to maintaining diversity
- Argmin: Faster local convergence but higher risk of local minima
