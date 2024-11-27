The attempt is based on Prof.ZHU's paper [A consensus-based global optimization method for high dimensional machine learning problemsA consensus-based global optimization method for high dimensional machine learning problems] (https://www.esaim-cocv.org/articles/cocv/pdf/2021/01/cocv190163.pdf)
# 1. Key observations from the comparison:

### Convergence Rate
The argmin method shows faster convergence to the optimal solution, as evidenced by the steeper descent in the convergence plot.

### Final Solution Quality:

The argmin method achieved a significantly better final solution  compared to the weighted average method. The argmin method's mean position is closer to the true optimum (0,0) for our test function
### Stability 
The convergence plot shows that the argmin method has a more stable descent path with fewer fluctuations compared to the weighted average method.

The superior performance of the argmin method in this case can be attributed to its direct selection of the best-performing particle in each batch, which provides a stronger optimization signal compared to the weighted average approach.

# 2.Theoretical properties

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
