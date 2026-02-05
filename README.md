RDL-MoE: A Risk-Stratified Deep Linear Mixture-of-Experts for Explainable Survival Analysis

This repository contains the official PyTorch implementation of the paper:  
A Risk-Stratified Deep Linear Mixture-of-Experts Architecture for Explainable Survival Analysis in Geriatric Lung Cancer, submitted to AHLI CHIL 2026.

[![Conference](https://img.shields.io/badge/CHIL-2026-blue)](https://www.ahli.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange)](https://www.python.org/)

---

Anonymity Statement
This repository is **anonymized** for the double-blind review process. All author names, affiliations, and institutional identifiers have been removed or masked.

---

Abstract
Precise prognostic stratification in elderly lung cancer patients is increasingly challenged by patient heterogeneity, wherein survival outcomes are dictated not only by tumor aggressiveness but also by the host’s physiological reserve. Current prognostic models fail to adequately capture this complexity. To bridge the gap between predictive accuracy and clinical explainability, we present the Risk-Stratified Deep Linear Mixture-of-Experts (RDL-MoE), a novel architecture explicitly designed to prioritize intrinsic explainability and safety. Departing from traditional tumor-centric approaches—and uniquely distinguishing our work from prior slice-based studies—we adopt a host-centric radiomics paradigm that leverages AI-driven automated segmentation to extract objective biomarkers from the volumetric extent of skeletal muscle, intramuscular adipose tissue, and the whole lung. Validated on multi-center cohorts, including the TCIA dataset, our model demonstrates predictive performance comparable to state-of-the-art baselines while offering substantially enhanced structural explainability. By harmonizing objective host-factor quantification with a safety-prioritized learning architecture, RDL-MoE provides a robust, transparent, and clinically actionable tool for survival prediction.

Repository Structure
├── main.py                # Entry point for training and evaluation
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation
├── src/
│   ├── config.py          # Hyperparameters and path configurations
│   ├── data_loader.py     # Data loading and ID matching logic
│   ├── preprocessing.py   # Feature selection (LASSO, RSF)
│   ├── models.py          # RDL-MoE model architecture (Experts & Gating)
│   ├── losses.py          # Cox loss and Ratio loss functions
│   └── utils.py           # Utility functions (Seed, Logging)
└── data/                  # Directory for datasets (created automatically)
