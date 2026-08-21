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

## 7. Worked Example: One Complete FedProx Round

This section walks through a full FL round with real numbers. Every calculation is shown explicitly so you can trace exactly what the algorithm does.

---

### 7.1 Server Settings Used in This Example

These are the values visible on the **Settings page** of the server dashboard (`http://localhost:8550`, Settings tab). They match the project defaults defined in `server/config.py`.

| Settings Page Field | Value | Where it comes from |
|---------------------|-------|---------------------|
| FL Rounds | 50 | `fl_rounds = 50` in `ServerSettings` |
| Local Epochs | 5 | `local_epochs = 5` in `ServerSettings` |
| **FedProx Mu** | **0.01** | `fedprox_mu = 0.01` in `ServerSettings` |
| DP Noise Sigma | 0.01 | `DP_NOISE_SIGMA=0.01` env var per client |
| Min Sites / Round | 3 | `min_sites_per_round = 3` in `ServerSettings` |
| Aggregation Mode | Quorum | Aggregate once ≥ 3 sites have submitted |
| Heartbeat Interval | 30 s | `heartbeat_seconds = 30` |
| Round Timeout | 300 s | `round_timeout_seconds = 300` |
| Learning Rate | 0.001 | `learning_rate = 0.001` (Adam optimiser) |

> **Note on μ in this example.** The production value is `μ = 0.01`. To make the proximal effect visible in the arithmetic below, this example uses `μ = 0.10`. The algorithm is identical — only the leash length changes. A callout box marks every place a number would differ at the real `μ = 0.01`.

---

### 7.2 The Five Sites: Data Snapshot

Each site has a private filtration dataset it never shares. The table below summarises the characteristics that drive each site's local training behaviour.

| Site | Filter type | TMP range (bar) | Dataset size (n) | Dominant Hermia regime | Character |
|------|-------------|-----------------|------------------|------------------------|-----------|
| Site 1 | PES 0.2 µm | 0.5 – 1.5 | **800** | Cake filtration | Largest; low-TMP cake conditions |
| Site 2 | PVDF 0.1 µm | 1.0 – 3.0 | **300** | Intermediate blocking | Mid-TMP, moderate fouling |
| Site 3 | Cellulose 0.45 µm | 0.3 – 0.8 | **500** | Combined 1-A | Low TMP, two-mechanism fouling |
| Site 4 | PES 0.1 µm | 1.5 – 4.0 | **150** | **Complete blocking** | **Smallest; extreme high-TMP outlier** |
| Site 5 | PVDF 0.22 µm | 0.8 – 2.0 | **600** | Standard blocking | Mid-range, typical conditions |
| **Total** | | | **2350** | | |

Site 4 is the interesting one. Its 150 high-TMP samples look nothing like the other four sites. Without FedProx, its local model will drift hard toward high-TMP-specific physics and contaminate the global model on aggregation.

---

### 7.3 Simplification: Two Weights Instead of ~50,000

The real PINN (`ParameterPredictor` in `shared/models/pinn.py`) has roughly **50,000 learnable parameters** across four linear layers (11→128→128→64→10). Tracking all of them in a worked example is impractical.

**For this example we reduce the entire PINN to two representative scalar weights:**

| Toy weight | Represents | Physical meaning |
|------------|-----------|-----------------|
| `J0_w` | Bias in the final layer that produces `J0` | Controls predicted initial flux (LMH) |
| `k1_w` | A weight connecting to the `k1` output | Controls pore-constriction rate in Combined 1-A model |

Everything else in the algorithm (proximal term, weighted aggregation, DP noise) works identically across all 50,000 real weights — we are only reducing the problem size, not changing any logic.

---

### 7.4 Round 1 — Step 1: Initial Global Model

At the start of Round 1 the server holds a randomly initialised global model. After reducing to our two toy weights:

```
W_global = { J0_w: 2.00,  k1_w: 0.50 }
```

The server broadcasts these weights to all five sites via `POST /federation/round/start`.

---

### 7.5 Round 1 — Step 2: Each Site's Local Optimum

Each site trains on its private data. Without any FedProx penalty, local training would push each site's weights toward the point that minimises its own local loss `Fₖ(W)`. Call this the **local optimum** for site k.

The local optima below are derived from the synthetic flux curves for each site's operating conditions — they represent where each site's data "wants" the model to go:

| Site | n | Local optimum `J0_w` | Local optimum `k1_w` | Why different? |
|------|---|----------------------|----------------------|----------------|
| Site 1 | 800 | 2.30 | 0.65 | High flux, cake regime → moderate J0 increase, higher k1 |
| Site 2 | 300 | 2.10 | 0.45 | Mid-TMP → small J0 increase, k1 barely changes |
| Site 3 | 500 | 2.25 | 0.70 | Two fouling mechanisms → higher k1 needed |
| Site 4 | 150 | **0.80** | **0.10** | **High TMP complete blocking → flux drops hard, k1 near zero** |
| Site 5 | 600 | 2.15 | 0.55 | Standard blocking, close to global start |

Notice that Sites 1, 2, 3, 5 all want `J0_w` somewhere in `[2.10, 2.30]` — they broadly agree. **Site 4 is a clear outlier**, pulling `J0_w` all the way down to `0.80`, because its high-TMP data sees catastrophic flux decline.

---

### 7.6 Round 1 — Step 3: Local Training WITH FedProx

FedProx prevents each site from fully reaching its local optimum. Instead, each site minimises the augmented loss:

```
h_k(W) = F_k(W)  +  (μ/2) × ‖W - W_global‖²
```

For a quadratic local loss (a good approximation near a minimum), the FedProx solution has a clean closed form. The proximal term shifts the optimum toward `W_global` by a factor of `μ/(1 + μ)`:

```
W_k_fedprox = ( W_k_local_opt  +  μ × W_global ) / ( 1 + μ )
```

With `μ = 0.10` and `W_global = {J0_w: 2.00, k1_w: 0.50}`:

**Site 1:**
```
J0_w = (2.30 + 0.10 × 2.00) / 1.10 = (2.30 + 0.20) / 1.10 = 2.50 / 1.10 = 2.273
k1_w = (0.65 + 0.10 × 0.50) / 1.10 = (0.65 + 0.05) / 1.10 = 0.70 / 1.10 = 0.636
```

**Site 2:**
```
J0_w = (2.10 + 0.20) / 1.10 = 2.30 / 1.10 = 2.091
k1_w = (0.45 + 0.05) / 1.10 = 0.50 / 1.10 = 0.455
```

**Site 3:**
```
J0_w = (2.25 + 0.20) / 1.10 = 2.45 / 1.10 = 2.227
k1_w = (0.70 + 0.05) / 1.10 = 0.75 / 1.10 = 0.682
```

**Site 4 — the outlier:**
```
J0_w = (0.80 + 0.10 × 2.00) / 1.10 = (0.80 + 0.20) / 1.10 = 1.00 / 1.10 = 0.909
k1_w = (0.10 + 0.10 × 0.50) / 1.10 = (0.10 + 0.05) / 1.10 = 0.15 / 1.10 = 0.136
```

**Site 5:**
```
J0_w = (2.15 + 0.20) / 1.10 = 2.35 / 1.10 = 2.136
k1_w = (0.55 + 0.05) / 1.10 = 0.60 / 1.10 = 0.545
```

**Summary — where each site ends up after local training:**

| Site | n | Without FedProx `[J0_w, k1_w]` | With FedProx μ=0.10 `[J0_w, k1_w]` | Site 4 drift saved |
|------|---|--------------------------------|--------------------------------------|--------------------|
| Site 1 | 800 | `[2.300, 0.650]` | `[2.273, 0.636]` | — |
| Site 2 | 300 | `[2.100, 0.450]` | `[2.091, 0.455]` | — |
| Site 3 | 500 | `[2.250, 0.700]` | `[2.227, 0.682]` | — |
| **Site 4** | **150** | **`[0.800, 0.100]`** | **`[0.909, 0.136]`** | **J0_w pulled 0.11 back toward 2.00** |
| Site 5 | 600 | `[2.150, 0.550]` | `[2.136, 0.545]` | — |
| W_global (start) | — | `[2.000, 0.500]` | `[2.000, 0.500]` | — |

Sites 1–3, 5 barely move — their local optima are already close to the global model so the proximal term barely fires. **Site 4 is the case that matters**: its unconstrained drift would have taken `J0_w` from `2.00` all the way to `0.80` (a gap of 1.20 units). FedProx reduced that to `0.909`, shrinking the drift to `1.09` units. Over 50 rounds, this compounding correction is what prevents the global model from being systematically pulled toward Site 4's extreme high-TMP conditions.

> **At μ = 0.01 (production setting):**
> Site 4's FedProx J0_w = (0.80 + 0.01×2.00) / 1.01 = 0.82/1.01 = **0.812**
> The correction is smaller per round but accumulates across 50 rounds.

---

### 7.7 Computing L_fedprox: What the Loss Function Sees

Inside `shared/models/pinn.py:filtration_loss()` (lines 430–439), the proximal penalty is evaluated every training step. Let's compute it for Site 4 at the END of local training (i.e., when the local weights have settled to their FedProx solution):

```python
# Site 4, with FedProx (μ = 0.10):
local_weights  = { "J0_w": 0.909, "k1_w": 0.136 }
global_weights = { "J0_w": 2.000, "k1_w": 0.500 }

prox  = (0.909 - 2.000)² + (0.136 - 0.500)²
      = (-1.091)²         + (-0.364)²
      = 1.190             + 0.133
      = 1.323

L_fedprox = (0.10 / 2) × 1.323 = 0.050 × 1.323 = 0.066
```

This `0.066` is added to Site 4's total loss `L_total = L_flux + L_LRV + L_physics + L_regime + 0.066`. The optimiser must pay this cost to drift far from `W_global` — so it only drifts as far as the reduction in `L_flux` justifies.

For comparison, Site 1 (which barely drifted):
```python
prox  = (2.273 - 2.000)² + (0.636 - 0.500)²
      = (0.273)²          + (0.136)²
      = 0.075             + 0.018
      = 0.093

L_fedprox = 0.050 × 0.093 = 0.005
```

Site 1 pays only `0.005` in proximal penalty. Site 4 pays `0.066` — **13× more** — reflecting how far its data pushed it from the global model.

> **At μ = 0.01 (production):**
> Site 4 L_fedprox = (0.01/2) × (0.812 - 2.00)² + (small k1 term) ≈ 0.005 × 1.411 ≈ **0.007**

---

### 7.8 Round 1 — Step 4: DP Noise Addition

Before transmitting, each site adds **Gaussian differential privacy noise** to its weight delta. This is configured via `DP_NOISE_SIGMA = 0.01` in each client's environment.

The delta is:
```
ΔW_k = W_k_fedprox - W_global
```

For Site 4:
```
ΔW_4[J0_w]  = 0.909 - 2.000 = -1.091
ΔW_4[k1_w]  = 0.136 - 0.500 = -0.364
```

Add noise sampled from N(0, 0.01²):
```
ΔW_4_noisy[J0_w]  = -1.091 + 0.003  = -1.088   (example noise draw: +0.003)
ΔW_4_noisy[k1_w]  = -0.364 + (-0.007) = -0.371  (example noise draw: -0.007)
```

The noise is tiny relative to the signal (magnitude ~0.01 vs delta magnitude ~1.09) — it provides meaningful privacy protection without degrading the update quality significantly. The server cannot tell if a delta of `-1.088` came from Site 4's true physics or from `-1.090 + 0.002` noise.

All five noisy deltas that arrive at the server:

| Site | n | ΔW[J0_w] (noisy) | ΔW[k1_w] (noisy) |
|------|---|-------------------|-------------------|
| Site 1 | 800 | +0.276 | +0.137 |
| Site 2 | 300 | +0.092 | -0.044 |
| Site 3 | 500 | +0.228 | +0.183 |
| Site 4 | 150 | **-1.088** | **-0.371** |
| Site 5 | 600 | +0.137 | +0.046 |

---

### 7.9 Round 1 — Step 5: Server Aggregation

The server (`server/core/aggregator.py:FedProxAggregator.aggregate()`) computes:

```
W_new[layer] = Σ_k  (n_k / N_total) × (W_global[layer] + ΔW_k_noisy[layer])
```

**Compute sample weights:**
```
N_total = 800 + 300 + 500 + 150 + 600 = 2350

w_1 = 800 / 2350 = 0.3404
w_2 = 300 / 2350 = 0.1277
w_3 = 500 / 2350 = 0.2128
w_4 = 150 / 2350 = 0.0638
w_5 = 600 / 2350 = 0.2553
```

**Aggregate J0_w:**
```
Each site's reconstructed weight = W_global[J0_w] + ΔW_k_noisy[J0_w]

Site 1:  2.00 + 0.276  = 2.276   ×  0.3404 = 0.775
Site 2:  2.00 + 0.092  = 2.092   ×  0.1277 = 0.267
Site 3:  2.00 + 0.228  = 2.228   ×  0.2128 = 0.474
Site 4:  2.00 + (-1.088)= 0.912  ×  0.0638 = 0.058
Site 5:  2.00 + 0.137  = 2.137   ×  0.2553 = 0.546
                                  ──────────────────
W_new[J0_w]                               = 2.120
```

**Aggregate k1_w:**
```
Site 1:  0.50 + 0.137  = 0.637   ×  0.3404 = 0.217
Site 2:  0.50 + (-0.044)= 0.456  ×  0.1277 = 0.058
Site 3:  0.50 + 0.183  = 0.683   ×  0.2128 = 0.145
Site 4:  0.50 + (-0.371)= 0.129  ×  0.0638 = 0.008
Site 5:  0.50 + 0.046  = 0.546   ×  0.2553 = 0.139
                                  ──────────────────
W_new[k1_w]                               = 0.567
```

**New global model after Round 1:**
```
W_global_round2 = { J0_w: 2.120,  k1_w: 0.567 }
```

---

### 7.10 What Would FedAvg Have Produced? (Head-to-Head)

Run the same aggregation but using the **unconstrained** local weights (no FedProx):

| Site | n | J0_w unconstrained | k1_w unconstrained |
|------|---|--------------------|--------------------|
| Site 1 | 800 | 2.300 | 0.650 |
| Site 2 | 300 | 2.100 | 0.450 |
| Site 3 | 500 | 2.250 | 0.700 |
| Site 4 | 150 | **0.800** | **0.100** |
| Site 5 | 600 | 2.150 | 0.550 |

```
FedAvg J0_w = 0.3404×2.30 + 0.1277×2.10 + 0.2128×2.25 + 0.0638×0.80 + 0.2553×2.15
            = 0.783 + 0.268 + 0.479 + 0.051 + 0.549
            = 2.130

FedAvg k1_w = 0.3404×0.65 + 0.1277×0.45 + 0.2128×0.70 + 0.0638×0.10 + 0.2553×0.55
            = 0.221 + 0.057 + 0.149 + 0.006 + 0.140
            = 0.573
```

**Round 1 outcome comparison:**

| Method | W_new[J0_w] | W_new[k1_w] | Site 4 influence |
|--------|-------------|-------------|-----------------|
| Starting global | 2.000 | 0.500 | — |
| **FedAvg** (no proximal) | 2.130 | 0.573 | Site 4 pulled J0_w down via its 0.80 local weight |
| **FedProx μ=0.10** | 2.120 | 0.567 | Site 4's outlier partially corrected to 0.909 |
| **FedProx μ=0.01** *(production)* | ≈ 2.129 | ≈ 0.572 | Very close to FedAvg this round — effect is cumulative |

The single-round difference looks small. That is expected — `μ = 0.01` is intentionally mild. The compounding benefit appears over 50 rounds: with FedAvg, Site 4's unconstrained updates repeatedly drag the global model toward high-TMP physics; with FedProx, each of Site 4's updates is partially corrected, and the global model converges to a point that generalises across all five TMP regimes rather than being biased by the outlier.

---

### 7.11 Round-by-Round Convergence Sketch

Below is a qualitative picture of how the two weights evolve across 10 rounds. The true optimum (what perfect global data would produce) is approximately `J0_w ≈ 2.18`, `k1_w ≈ 0.58`.

```
Round:       0     1     2     3     4     5     6     7     8     9    10
─────────────────────────────────────────────────────────────────────────
FedAvg J0_w: 2.00  2.13  2.14  2.15  2.16  2.16  2.16  2.17  2.17  2.17  2.17
FedProx J0_w:2.00  2.12  2.14  2.15  2.16  2.17  2.17  2.18  2.18  2.18  2.18
True optimum:                                                          2.18

FedAvg k1_w: 0.50  0.57  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58
FedProx k1_w:0.50  0.57  0.57  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58
True optimum:                                                          0.58
```

Key observations:
- **FedProx converges to the true global optimum**; FedAvg plateaus slightly below it because Site 4's unconstrained drift creates a persistent bias.
- Both converge in roughly the same number of rounds — FedProx does not slow things down, it corrects the destination.
- In more heterogeneous scenarios (e.g. if Sites 3 and 4 were swapped in size, making the outlier dominant), FedAvg would diverge and FedProx would remain stable.

---

### 7.12 Putting It All Together: The Full Round Lifecycle

```
Server (http://localhost:8550)
│
│  Settings Page:  FL Rounds=50, Local Epochs=5, FedProx Mu=0.01,
│                  DP Noise Sigma=0.01, Min Sites=3, Aggregation=Quorum
│
│  Round 1 starts
│  ─────────────────────────────────────────────────────────
│  BROADCAST  W_global={J0_w:2.00, k1_w:0.50}  →  all 5 sites
│
├─ Site 1 (n=800, cake)          local train 5 epochs → ΔW=[+0.276, +0.137] + DP noise
├─ Site 2 (n=300, intermediate)  local train 5 epochs → ΔW=[+0.092, -0.044] + DP noise
├─ Site 3 (n=500, combined 1-A)  local train 5 epochs → ΔW=[+0.228, +0.183] + DP noise
├─ Site 4 (n=150, complete)      local train 5 epochs → ΔW=[-1.088, -0.371] + DP noise
│                                  ^ FedProx pulled this from [-1.20, -0.40]
└─ Site 5 (n=600, standard)      local train 5 epochs → ΔW=[+0.137, +0.046] + DP noise
│
│  Quorum reached (5 ≥ 3 min sites)
│  AGGREGATE  weighted average by n_k / N_total
│  ─────────────────────────────────────────────────────────
│  W_global_round2 = {J0_w: 2.120,  k1_w: 0.567}
│
│  Emit structured audit log entry for Round 1
│  Broadcast W_global_round2  →  Round 2 begins
```

---

## 6. References

1. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). **Federated Optimization in Heterogeneous Networks.** *Proceedings of Machine Learning and Systems (MLSys)*. arXiv:1812.06127.

2. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., & Suresh, A. T. (2021). **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.** *ICML*. arXiv:1910.06378.

3. Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). **Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization.** *NeurIPS*. arXiv:2007.07481.

4. Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saligrama, V. (2021). **Federated Learning Based on Dynamic Regularization.** *ICLR*. arXiv:2111.04263.

5. Li, Q., He, B., & Song, D. (2021). **Model-Contrastive Federated Learning.** *CVPR*. arXiv:2103.16257.

---

*All source code references are to `D:\viral_fl_project`. Implementation of FedProx: `shared/models/pinn.py:filtration_loss()` lines 430–439. Aggregation: `server/core/aggregator.py:FedProxAggregator.aggregate()`.*
