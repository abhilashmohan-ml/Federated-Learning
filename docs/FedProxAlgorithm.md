# FedProx in the Viral Filtration Federated Learning System

**Document version:** 1.0 · August 2026
**Scope:** Algorithm description, mathematical formulation, and alternative algorithms for the Viral Filtration FL project

---

## 1. Abstract

FedProx (Li et al., 2020, *Federated Optimization in Heterogeneous Networks*, MLSys 2020) is a federated learning algorithm that extends FedAvg by adding a **proximal regularisation term** to each site's local loss during training. The proximal term prevents local model weights from drifting too far from the current global model, enabling stable and theoretically guaranteed convergence even when participating sites have **heterogeneous data distributions** (non-IID) and **unequal computational capacity**. In this project, FedProx is implemented as an additional loss component (`L_fedprox`) in `shared/models/pinn.py:filtration_loss()`, with the aggregation step remaining identical to FedAvg in `server/core/aggregator.py`. The strength of the constraint is governed by the hyperparameter `μ` (configured via `FEDPROX_MU`, defaulting to `0.01`).

---

## 2. Conceptual Explanation

### 2.1 The Starting Point: Why Not Just Use FedAvg?

The simplest federated learning algorithm — **FedAvg** — works as follows:

1. Server sends the current global model weights `W_global` to all sites.
2. Each site trains on its own local data for `LOCAL_EPOCHS` epochs, producing updated weights `W_local`.
3. Server collects all `W_local` values and averages them (weighted by site dataset size) to produce new `W_global`.

This is exactly what `server/core/aggregator.py:FedProxAggregator.aggregate()` implements. The aggregation formula is:

```
W_new = Σᵢ (nᵢ / N_total) × (W_global + ΔWᵢ)
```

**The problem with FedAvg in our context:**

Our five manufacturing sites produce very different data. Consider:

| Site | Filter type | TMP range (bar) | Data volume | Dominant regime |
|------|-------------|-----------------|-------------|-----------------|
| Site 1 | PES 0.2 µm | 0.5–1.5 | 8,000 samples | Cake filtration |
| Site 2 | PVDF 0.1 µm | 1.0–3.0 | 3,000 samples | Intermediate blocking |
| Site 3 | Cellulose 0.45 µm | 0.3–0.8 | 5,000 samples | Combined 1-A |
| Site 4 | PES 0.1 µm | 1.5–4.0 | 1,500 samples | Complete blocking |
| Site 5 | PVDF 0.22 µm | 0.8–2.0 | 6,000 samples | Standard blocking |

Site 4 has only 1,500 samples in a very different TMP regime. If Site 4 trains for 5 local epochs, its model will overfit heavily to high-TMP, complete-blocking conditions and diverge significantly from the global model. When this diverged update is averaged in, it degrades the global model for everyone.

In federated learning jargon, this is called **client drift**: local models drift away from the global optimum because each site's gradient direction is biased toward its own local data distribution.

### 2.2 The FedProx Fix: A Leash on Local Training

FedProx solves client drift by adding a **proximal term** — think of it as a "leash" that keeps each site's local model tethered to the global model during training.

**Analogy:** Imagine a dog (Site 4's model) on a leash attached to a post (the global model). The dog can still move and explore its local territory (train on local high-TMP data), but the leash prevents it from running so far that it can no longer return to the post quickly. The length of the leash is `μ`.

In code (`shared/models/pinn.py:filtration_loss`, lines 430–439):

```python
L_fedprox = torch.tensor(0.0)
if global_weights is not None and local_weights is not None:
    prox = sum(
        torch.sum((local_weights[k] - global_weights[k]) ** 2)
        for k in global_weights
        if k in local_weights
    )
    L_fedprox = (fedprox_mu / 2.0) * prox
```

The total loss every site minimises becomes:

```
L_total = L_flux + L_LRV + L_physics + L_regime + L_fedprox
```

`L_fedprox` is non-zero as soon as `W_local ≠ W_global`. Every gradient update step simultaneously tries to fit the local data AND stay close to the global model. These two objectives compete — the hyperparameter `μ` controls the balance.

### 2.3 The Role of μ (FEDPROX_MU)

| μ value | Behaviour | Use case |
|---------|-----------|----------|
| `0.0` | Identical to FedAvg — no leash | All sites have identical data (IID) |
| `0.01` (our default) | Mild constraint — sites can adapt but drift is limited | Moderate heterogeneity (our scenario) |
| `0.1` | Strong constraint — local models stay very close to global | Highly heterogeneous data, risk of divergence |
| `1.0` | Very strong — almost no local adaptation | Sites are unreliable or data is wildly different |

**Concrete example for our project:**

Suppose at round 10, Site 4's `L_flux` gradient wants to push parameter `ks` (standard blocking rate) from `0.05` towards `0.001` (because high TMP almost eliminates standard blocking at Site 4). The current global `ks = 0.04`. With `μ = 0.01`:

- `L_flux` gradient pushes `ks` strongly down.
- `L_fedprox` gradient pushes `ks` back toward `0.04` with force proportional to `(0.001 - 0.04)² × 0.01`.
- The net update moves `ks` down but not as far as Site 4's data alone would dictate.

After aggregation, the global model still reflects the average physics across all sites, not just the outlier Site 4.

### 2.4 What the Aggregation Server Does (Unchanged from FedAvg)

A key architectural fact: **the aggregation in `aggregator.py` is identical to FedAvg**. FedProx is purely a client-side change. The server simply:

1. Receives weight deltas `ΔW` from each site.
2. Computes a sample-weighted average.
3. Broadcasts the new global model.

The proximal term does its work silently inside each site's `filtration_loss()` during local training. The server never sees or checks the proximal term.

---

## 3. Mathematical Formulation

### 3.1 Problem Setup

Let there be `K` sites (in our case, K = 5). Each site `k` has a local dataset `D_k = {(xᵢ, yᵢ)}` of filtration measurements, where:
- `xᵢ` = feature vector (TMP, feed concentration, filter geometry, virus properties) ∈ ℝ¹¹
- `yᵢ` = labels (observed flux curve `J(t)`, LRV, Hermia regime)

Define the **local objective** at site `k` as:

```
Fₖ(W) = (1/|Dₖ|) Σᵢ∈Dₖ  ℓ(W; xᵢ, yᵢ)
```

where `ℓ` is the sum of `L_flux + L_LRV + L_physics + L_regime` from `filtration_loss()`.

The **global objective** is the sample-weighted aggregate:

```
F(W) = Σₖ (nₖ / N) × Fₖ(W)
```

where `nₖ = |Dₖ|` and `N = Σₖ nₖ`.

**Goal:** Find `W* = argmin F(W)`.

### 3.2 FedAvg Local Subproblem (Baseline)

In standard FedAvg, each site `k` in round `t` solves:

```
W_k^(t+1) = argmin_{W}  Fₖ(W)
```

starting from `W_global^(t)` using SGD for `E` local epochs. This has **no constraint** — the site can converge to any local minimum, which may be far from the global optimum.

### 3.3 FedProx Local Subproblem

FedProx changes the local subproblem by adding the proximal term:

```
W_k^(t+1) = argmin_{W}  hₖ(W; W_global^(t))
```

where:

```
hₖ(W; W_global^(t))  =  Fₖ(W)  +  (μ/2) × ‖W - W_global^(t)‖²
```

**Term-by-term meaning:**

| Term | Symbol | Meaning |
|------|--------|---------|
| Local data fit | `Fₖ(W)` | Minimise prediction error on local filtration data |
| Proximal regulariser | `(μ/2) ‖W - W_global‖²` | Penalise deviation from the global model |
| Regularisation strength | `μ` | Controls the leash length (our default: `0.01`) |

The `‖·‖²` is the squared L2 norm summed over all weight parameters — in our PINN, this spans all layers of `ParameterPredictor` and `BlockingRegimeClassifier`.

Expanded to individual weight layers `θ_l` (what `filtration_loss()` computes on lines 434–438):

```
(μ/2) × ‖W_local - W_global‖²  =  (μ/2) × Σ_l  Σ_j  (θ_l,j^local - θ_l,j^global)²
```

### 3.4 Gradient of the Proximal Term

During backpropagation, the gradient of the proximal term with respect to weight `θ` is:

```
∂/∂θ  [(μ/2) × (θ - θ_global)²]  =  μ × (θ - θ_global)
```

This is simply a **linear gradient** pointing from the local weight back toward the global weight. It acts like a spring force — the further `θ` has moved from `θ_global`, the stronger the pull back.

### 3.5 Global Aggregation (FedAvg-identical)

After all participating sites complete local training in round `t`:

```
W_global^(t+1)  =  Σ_k  (nₖ / N) × W_k^(t+1)
```

In delta form (as implemented in `aggregator.py`):

```
W_global^(t+1)  =  Σ_k  (nₖ / N) × (W_global^(t) + ΔWₖ)
```

where `ΔWₖ = W_k^(t+1) - W_global^(t)` is the weight update transmitted by site `k`.

### 3.6 Convergence Guarantee

Li et al. (2020) prove that under **non-convex** objectives (which applies to our PINN), FedProx converges to a neighbourhood of a stationary point when:

- `γ-inexact` local solutions are used (sites don't have to fully minimise `hₖ`, just get within a factor `γ ∈ [0,1]` of the optimum)
- `μ` satisfies a bound related to the **dissimilarity** between local and global gradients

The convergence bound is:

```
(1/T) Σ_{t=1}^{T}  E[‖∇F(W_global^(t))‖²]  ≤  B(μ, γ, K, E)
```

where `B` decreases with larger `μ` (stronger proximal constraint) and smaller gradient dissimilarity. FedAvg (μ=0) has no such guarantee under non-IID data.

---

## 4. Alternative and Newer Algorithms

The federated learning landscape has moved quickly since FedProx (2020). Below are the most relevant alternatives for this system, from most to least directly applicable.

---

### 4.1 SCAFFOLD — Stochastic Controlled Averaging
**Paper:** Karimireddy et al., ICML 2020 (arXiv:1910.06378)

**Core idea:** Instead of penalising local drift with a proximal term (FedProx), SCAFFOLD **corrects gradient directions** using **control variates** — per-site variance-reduction signals that tell each site: "your local gradient is biased by *this much* relative to the global gradient; subtract it."

**How it differs from FedProx:**

| Aspect | FedProx | SCAFFOLD |
|--------|---------|---------|
| Mechanism | Penalise weight distance | Correct gradient direction |
| Extra communication | None | One control variate vector per site per round |
| Convergence | Bounded by gradient dissimilarity | Not affected by data heterogeneity |
| Communication rounds needed | More (heterogeneity slows it) | Fewer (bias corrected directly) |

**Applicability to this project:**

SCAFFOLD would be a strong upgrade for rounds where Sites 4 and 5 (smallest datasets) currently diverge. The control variate for each site is a vector the same size as `W`, computed and stored on the server. Extra memory and communication cost are modest (one extra `W`-sized message per site per round).

**Implementation change needed:** Add control variate state (`cₖ`, `c`) to `RoundManager` and `FLClient`; modify `local_trainer.py` gradient update step; modify `aggregator.py` to also update control variates.

---

### 4.2 FedNova — Federated Normalised Averaging
**Paper:** Wang et al., NeurIPS 2020 (arXiv:2007.07481)

**Core idea:** FedAvg and FedProx both suffer from **objective inconsistency** — when sites perform different numbers of local gradient steps, the effective optimisation target shifts away from the intended global objective. FedNova fixes this by **normalising** each site's update by the number of local steps taken.

**The inconsistency problem in our context:**

Imagine Site 1 (8,000 samples) completes 5 local epochs = 80 gradient steps. Site 4 (1,500 samples) completes 5 local epochs = 15 gradient steps. When their updates are averaged, Site 1 has effectively taken much larger steps. The aggregated gradient does not correspond to *any* global loss function.

FedNova computes:

```
d_k  =  (1/aₖ) × Σ_{local steps}  gradient_step_k
```

where `aₖ` is a **normalisation factor** (typically the effective number of local steps). The aggregation then takes:

```
W_global^(t+1)  =  W_global^(t)  -  τ_eff × Σ_k (nₖ/N) × d_k
```

This makes convergence analysis exact regardless of per-site step heterogeneity.

**Applicability:** In our setup, all sites use the same `LOCAL_EPOCHS=5`, but their dataset sizes differ by ~5×, so the number of gradient steps per epoch differs substantially. FedNova would directly address this systematic bias.

---

### 4.3 FedDyn — Federated Learning via Dynamic Regularization
**Paper:** Acar et al., ICLR 2021 (arXiv:2111.04263)

**Core idea:** Rather than using a *static* proximal term tethered to `W_global` (FedProx), FedDyn introduces a **dynamic, per-client, per-round regulariser** that is updated each round to track the gradient history. This ensures that, in the limit, each site's local minimum aligns exactly with the global minimum — not just gets "close" to it.

**Local objective per site `k`, round `t`:**

```
hₖ^(t)(W)  =  Fₖ(W)  -  〈∇Fₖ(W_k^(t-1)), W〉  +  (α/2) × ‖W - W_global^(t)‖²
```

The middle term `−〈∇Fₖ(W_k^(t-1)), W〉` is the history correction — it records how this site's gradient diverged in the previous round and counteracts it in the current round.

**Why better than FedProx:**

- FedProx: convergence holds but admits a residual bias proportional to `μ` and gradient dissimilarity.
- FedDyn: the dynamic correction drives the bias to **zero** asymptotically — local solutions provably converge to the exact global solution.

**Cost:** Each site must store and transmit one extra gradient-history vector per round. Overhead is similar to SCAFFOLD.

---

### 4.4 MOON — Model-Contrastive Federated Learning
**Paper:** Li, He, Song, CVPR 2021 (arXiv:2103.16257)

**Core idea:** Rather than penalising weight-space distance (FedProx) or correcting gradient directions (SCAFFOLD), MOON applies **contrastive learning at the representation level**. Each site trains by:

1. Pulling its representation layer *closer* to the global model's representations.
2. Pushing its representation layer *away* from its own previous-round (stale) representations.

**Why relevant for our PINN:**

Our `ParameterPredictor` maps raw features to latent physical parameters `{J0, ks, ki, ...}`. This latent space *is* a representation. MOON's contrastive objective would:
- Keep the physical parameter predictions aligned across sites in representation space (not just weight space).
- Actively discourage stagnation in local representation modes (e.g., Site 4 getting stuck predicting `ks ≈ 0` for all inputs).

**Limitation:** MOON requires storing previous-round local model weights at each site, and the contrastive loss requires a batch of same-input representations from both the global and local-previous model at training time — modestly more complex to implement.

---

### 4.5 Summary Comparison Table

| Algorithm | Year | Drift mechanism targeted | Extra communication | Convergence guarantee | Best fit for this project |
|-----------|------|--------------------------|---------------------|----------------------|---------------------------|
| **FedAvg** | 2017 | None | None | Only for IID data | Baseline only |
| **FedProx** ✓ (current) | 2020 | Weight distance penalised | None | Non-IID, non-convex | Currently in use |
| **SCAFFOLD** | 2020 | Gradient direction corrected | +1 control variate per site | Not affected by heterogeneity | **Best candidate to replace FedProx** |
| **FedNova** | 2020 | Objective inconsistency from unequal steps | None | Exact for normalised steps | **Complementary to FedProx — easy add** |
| **FedDyn** | 2021 | Dynamic per-round correction | +1 gradient history per site | Exact global convergence | Strong upgrade, moderate complexity |
| **MOON** | 2021 | Representation-level drift | +1 previous model stored per site | Empirically strong on deep nets | Interesting for PINN representation layer |

---

## 5. Recommendations for This Project

**Short term (low implementation cost):**

Combine **FedProx + FedNova normalisation**. The normalisation fix is purely in the aggregation step (`aggregator.py`) — divide each site's weight delta by its number of local gradient steps before aggregating. This costs zero extra communication and directly addresses the 5× dataset size difference across our sites.

**Medium term (moderate complexity):**

Replace FedProx with **SCAFFOLD**. The control variate idea is mathematically cleaner than the proximal penalty and provably convergent regardless of data heterogeneity. Implementation requires:
- Server stores one control variate vector `c` (same size as `W`).
- Each site stores its own `cₖ`, updated each round.
- An extra `W`-sized upload per round — in our FL setup with small PINN weights (~50K parameters × 4 bytes = ~200 KB), this is negligible.

**If physics-informed aspects become the bottleneck:**

The proximal term penalises *all weights equally*. A physics-informed extension would weight the proximal term differently for physics-grounded parameters (`J0`, `ks`, etc.) vs. purely learned embedding layers. This is novel enough to be worth a paper but is speculative relative to the immediate engineering goals.

---

## 6. References

1. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). **Federated Optimization in Heterogeneous Networks.** *Proceedings of Machine Learning and Systems (MLSys)*. arXiv:1812.06127.

2. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2021). **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.** *ICML*. arXiv:1910.06378.

3. Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). **Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization.** *NeurIPS*. arXiv:2007.07481.

4. Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saligrama, V. (2021). **Federated Learning Based on Dynamic Regularization.** *ICLR*. arXiv:2111.04263.

5. Li, Q., He, B., & Song, D. (2021). **Model-Contrastive Federated Learning.** *CVPR*. arXiv:2103.16257.

---

*All source code references are to `D:\viral_fl_project`. Implementation of FedProx: `shared/models/pinn.py:filtration_loss()` lines 430–439. Aggregation: `server/core/aggregator.py:FedProxAggregator.aggregate()`.*
