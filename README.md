# A Comparative Study of BVARs with Non-Gaussian Innovations

This repository hosts the  project for **EN.553.724 Probabilistic Machine Learning (Spring 2026)** at Johns Hopkins University.

**Authors:** Jonathan Ma, Sijia Zhu, Sibo Tao
**Instructor:** Dr. Holden Lee

---

## Project Overview

This project develops and evaluates robust probabilistic modeling for multivariate financial time series using Bayesian Vector Autoregressions (BVARs). A BVAR can be viewed as a generative graphical model specifying a joint distribution and latent parameters, where uncertainty arises from both innovation randomness and parameter uncertainty. Standard BVAR implementations assume Gaussian innovations. However, empirical evidence in macro-financial data suggests heavy tails, volatility clustering, and tail-risk behavior. From a probabilistic machine learning perspective, this constitutes distributional misspecification in the generative model.

---

## Data Source

**Dataset:** TBD

---

## Methodology

TBD

---


## Structure

```text
pmlproject-MaZhuTao-SP26/
├── code/                         # Analysis scripts and RMarkdown modules
├── data/
│   └── raw/                      # Immutable input datasets
│   └── cleaned/                  # Versions of cleaned data
├── docs/                         # Documentation and team coordination
│   ├── CHANGELOG.md              # Project updates and version history
│   ├── ChangeTracking.md          # Collaboration guidelines and author credits
│   └── InitialSetup.md                  # Environment setup and package requirements
├── Reports/                      # Knitted and polished Files
│   └── Report.pdf                # Final Report
├── .gitignore                    # Files and folders excluded from Git tracking
├── LICENSE                       # Usage license
└── README.md                     # Project overview 
```
