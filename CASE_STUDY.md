# Case Study: Causal override for dynamic surge

## Problem

Marketplace pricing systems (taxi / delivery) often apply **surge** or discounts with rule-based or predictive models that estimate *level* of acceptance or conversion:

\[
P(Y=1 \mid X, T)
\]

That confuses three groups:

| Group | Behavior | If you always treat |
|:---|:---|:---|
| **Persuadables** | Convert *because* of treatment | Correct spend |
| **Sure things / lost causes** | Same outcome with or without treatment | Wasted incentive / noise |
| **Sleeping dogs** | Treatment *hurts* acceptance | Active damage (churn, reject) |

In a companion **Dynamic Pricing Engine (DPE)** simulation, additive vs multiplicative surge is A/B-tested. The open question for portfolio work: **who should not receive aggressive surge at all?**

## Approach

I built **Causal Pricing Engine (CPE)** as a small service next to DPE:

1. **Target** — Individual Treatment Effect (uplift):
   \[
   \tau(x)=\mathbb{E}[Y^{(1)}-Y^{(0)}\mid X=x]
   \]
2. **Model** — T-Learner (two outcome models) on sklearn GBM; optional S/X/DML remain in the codebase but are not required for the demo path.
3. **Data** — synthetic data with seeded heterogeneous TE for a reproducible story, plus a connector to DPE `simulation_analytics` SQLite for the same feature schema.
4. **Serve** — FastAPI `POST /predict_uplift` on `:8100`.
5. **Integrate** — DPE calls CPE with timeout ≤200 ms, **fail-open** if CPE is down; if \(\hat\tau(x) < -\theta\) (default \(\theta=0.05\)), DPE sets surge to base fare (`CAUSAL_NO_SURGE`) and exposes `causal_*` fields on the price response.
6. **Parity** — `past_trips` / `avg_surge` are defined *as-of prior trip* offline (`load_dpe_data`) and online (`DriverHistoryStore`).

## What “good” means here

| Signal | Role |
|:---|:---|
| **Qini coefficient / Uplift@k** | Offline ranking quality of \(\hat\tau\) (Qini is **normalized** ≈[-1,1], not raw area) |
| **Honest demo** | Labels *persuadable / neutral / sleeping dog* come from real scored rows, not hand-waved features |
| **Portfolio proof** | Live HTTP scores + the same override rule DPE uses, captured in `docs/evidence/latest_proof.md` |

This is **not** a claim of production lift on a city-scale A/B. Simulation n is small; experiment power for raw conversion deltas is limited (see archived CUPED eval). The portfolio claim is narrower and honest:

> Given HTE in the data, we can score ITE, map it to a pricing policy, and wire that policy into a pricing service without taking the pricing path down when the model is unavailable.

> **Methodological & Pipeline Disclaimers:**
> - **Synthetic DGP**: Uses a synthetic dataset with **seeded Heterogeneous Treatment Effects (HTE)** where true \(\tau(x)\) is a piecewise function of `past_trips` and `surge_bonus`. The demo proves **end-to-end pipeline execution and policy integration**, not actual city-scale production lift.
> - **Normalized Qini**: Evaluated metrics report the **normalized Qini coefficient** (scale \([-1, 1]\), where random ranking \(\approx 0\) and oracle \(\approx 1\)), avoiding legacy unnormalized \(O(n^2)\) area misinterpretations.
> - **DPE Observational Data**: Training models on DPE switchback simulation logs uses observational data where post-treatment features (such as `price` and `surge_bonus`) may be treatment-entangled. Pre-treatment feature sets are provided for causal purity during offline training (see Stage C / `CAUSAL.md`).

## Trade-offs I accepted

| Choice | Why |
|:---|:---|
| Fail-open | Wrong surge is better than no price / timeout cascade |
| Hard threshold on \(\hat\tau\) | Interpretable Sleeping Dog rule; easy to audit |
| Separate CPE process | Shows integration (timeout, schema, ops) not only a notebook |
| Synthetic default for demo | Guarantees +/− HTE so labels stay true; DPE path via `--source dpe` |

## How to reproduce (5 minutes)

```powershell
cd causal-pricing-engine
pip install -r requirements.txt
pip install -e .
python scripts/demo.py              # honest three roles
python scripts/portfolio_proof.py   # HTTP + evidence markdown
```

Optional dual stack: CPE `:8100`, DPE `:8000`, then `python scripts/portfolio_proof.py --with-dpe`.

## What I would do next (if this were a product)

1. Larger switchback logs and CUPED-powered readouts of the override policy.  
2. Multi-treatment (surge levels / discount depths), not binary treat/control.  
3. Shared feature store so online graph features match offline training columns under load.

## One-line summary

**Uplift scoring as a fail-open sidecar that stops surge for predicted Sleeping Dogs — demonstrated with honest holdout picks and captured HTTP evidence, integrated with a graph-based dynamic pricing simulator.**
