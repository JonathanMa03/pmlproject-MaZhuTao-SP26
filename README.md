# A Comparative Study of BVARs with Non-Gaussian Innovations

This repository hosts the final project for **EN.553.724 Probabilistic Machine Learning (Spring 2026)** at Johns Hopkins University.

**Authors:** [Jonathan Ma](https://jonathanma03.github.io/), [Sijia Zhu](https://orcid.org/0009-0006-3694-6473), Sibo Tao  
**Instructor:** Dr. Holden Lee

---

## Project Overview

This project develops and evaluates a sequence of increasingly flexible Bayesian Vector Autoregression (BVAR) models for multivariate financial time series. From a probabilistic machine learning perspective, a BVAR defines a generative model over observed returns and latent parameters, where uncertainty arises from both innovation noise and posterior parameter uncertainty.

Standard BVAR implementations assume Gaussian innovations. However, empirical financial data exhibits:

- Heavy tails (extreme events)
- Volatility clustering (time-varying variance)
- Regime shifts (structural breaks)

These phenomena imply **distributional misspecification** under Gaussian assumptions. This project addresses this limitation by introducing and comparing non-Gaussian extensions that better capture real-world dynamics.

---

## Data Source

We construct a multivariate dataset of daily log returns for major U.S. equity ETFs:

- SPY (S&P 500)
- QQQ (NASDAQ)
- DIA (Dow Jones)
- IWN (Russell 2000 Value)

### ETL Pipeline
- Align trading dates across assets
- Compute daily log returns
- Standardize series for numerical stability

Exploratory analysis confirms:
- Strong cross-asset correlation
- Heavy-tailed return distributions
- Clear volatility clustering

---

## Methodology

We implement and compare four models of increasing complexity:

### 1. Gaussian VAR (Baseline)
- Classical VAR with Gaussian innovations
- Estimated via OLS
- Captures linear dependence and correlation structure
- **Limitation:** fails under heavy tails and heteroskedasticity

---

### 2. Student-$t$ VAR (Heavy-Tailed Innovations)
- Scale-mixture representation with latent $\lambda_t$
- Downweights extreme observations adaptively
- Handles heavy-tailed behavior

**Key Insight:** Standardized residuals become approximately Gaussian after scaling.

---

### 3. Mixture VAR (Regime Switching)
- Two-component mixture of Gaussian VARs
- Captures:
  - Low-volatility regime (dominant)
  - High-volatility regime (crisis periods)

**Key Insight:** Regime probabilities align with market stress events (e.g., 2008, 2020).

---

### 4. Stochastic Volatility VAR (SV-VAR)
- Time-varying latent log-volatility following AR(1)
- Estimated via MCMC with Metropolis updates
- Captures continuous volatility evolution

**Key Insight:** Latent volatility closely tracks absolute returns.

---

## Inference and Diagnostics

All Bayesian models are estimated using MCMC.

We evaluate:
- Trace plots (mixing behavior)
- Effective Sample Size (ESS)
- $\hat{R}$ convergence diagnostics

Findings:
- Student-$t$ model exhibits strong mixing
- Mixture model has lower ESS for regime-specific covariance (expected)
- SV model shows stable acceptance rates and convergence

---

## Predictive Evaluation

### In-Sample
We compare one-step predictive distributions:
- Gaussian underestimates tail risk
- Student-$t$ improves tail fit
- Mixture captures extreme outcomes most effectively

---

### Out-of-Sample (Rolling Forecasts)

We evaluate models using:
- Log Predictive Score (LPS)
- Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Predictive variance

#### Results Summary:
- **Mixture VAR** achieves best predictive performance (highest LPS)
- **Student-$t$ VAR** performs second best
- **Gaussian VAR** and **SV-VAR** lag slightly

---

## Key Results

- Heavy tails significantly impact predictive accuracy  
- Regime-switching models provide the best overall fit  
- Stochastic volatility captures dynamics but is less competitive in LPS  
- Gaussian assumptions are strongly rejected by the data  

---

## Visual Analysis

The notebook includes:

- Return time series and volatility clustering
- Residual diagnostics (Gaussian vs Student-$t$)
- Latent volatility trajectories (SV model)
- Regime probabilities (Mixture model)
- Predictive distribution comparisons
- Rolling and cumulative log predictive scores
- Tail risk comparisons (VaR and ES)

---

## Conclusion

This project demonstrates that incorporating **non-Gaussian innovations** is critical for realistic financial time series modeling. Each model addresses a specific limitation:

| Model        | Strength                          | Weakness                         |
|--------------|----------------------------------|----------------------------------|
| Gaussian     | Simple, interpretable            | Misspecified tails & volatility  |
| Student-$t$  | Handles heavy tails              | No regime structure              |
| Mixture      | Captures regime shifts           | More complex inference           |
| SV           | Models time-varying volatility   | Less competitive predictive fit  |

**Final takeaway:**  
The mixture VAR provides the strongest overall performance by combining flexibility with explicit modeling of structural changes in the data.

---

## Structure

```text
pmlproject-MaZhuTao-SP26/
├── code/                         # Model implementations and evaluation scripts
├── data/                         # Data used for analysis from Sibo
│   ├── raw/                      # Immutable input datasets
│   └── processed/                # Placeholder for processed data
├── docs/                         # Documentation and collaboration notes
│   ├── CHANGELOG.md
│   ├── ChangeTracking.md
│   └── InitialSetup.md
├── outputs/                      # Outputs from code, indexed by date
├── reports/                      # Final report
│   └── Report.pdf
├── src/                          # scripts and source
├── config.py                     # Directory control
├── models_notebook.ipynb         # Analysis notebook
├── .gitignore
├── LICENSE
└── README.md
```

