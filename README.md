
# SwinBayesNet

**SwinBayesNet** is a deep learning model designed for searching hot subdwarf stars from SDSS images, combining **Swin Transformer** for feature extraction and **Bayesian Neural Networks** for classification and uncertainty estimation. 

## Contents

- **Backbone: Feature Extraction**: `Swin_transformer_model.py` contains the feature extraction for SDSS images (5 bands). We provide two models: one for **two-class classification** and another for **three-class classification**.
- **Head: Classification + Uncertainty**: In the **Bayesian_model** folder, `Bayesian_head.py` contains the head for **SwinBayesNet**. It provides two models for **two-class classification** (1024 features) and **three-class classification** (1536 features).

## File Structure

```
.
├── Bayesian_model
│   ├── __init__.py             # Initialization file for the Bayesian model
│   ├── metrics.py             # Metrics related to the model evaluation
│   ├── misc.py                # Utility functions and helpers
│   ├── BBB_LRT_BBBLinear.py   # Custom Bayesian Linear Layer (LRT)
│   ├── Bayesian_head.py       # Bayesian head for SwinBayesNet (classification + uncertainty)
├── Swin_transformer_model.py  # Swin Transformer backbone for feature extraction
└── README.md                  # Project overview and instructions
```
