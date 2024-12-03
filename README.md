# SwinBayesNet

SwinBayesNet is a Deep
Learning Method which is for Search for Hot Subdwarf Stars from SDSS Images, which combining the Swin Transformer and Bayesian Neural Networks. The model is initially trained for two-class and three-class classification using the Swin Transformer. Once the model is trained stably, we freeze the backbone (feature extraction) and replace the output head with a Bayesian neural network to produce the final classification results and uncertainty.The more details are seen in paper.

## Contents

- **Backbone: Feature Extraction**: `Swin_transformer_model.py` contains the feature extraction for SDSS Images (5 bands), we have provides two-stage model for two-class classification and two-class classification model.
- **Head: Classification+Uncertainty**: In the Bayesian_model floder, `Bayesian_head.py` contains the head for SwinBayesNet, we have provides two-stage model for two-class classification (1024 features) and two-class classification (1536) model.

```

## File Structure

```
├── swin_two_classification.py        # Swin Transformer model for binary classification
├── swin_three_classification.py      # Swin Transformer model for ternary classification
├── bayesian_head.py                  # Bayesian head for uncertainty estimation
├── BBB_LRT_BBBLinear.py              # Custom Bayesian Linear Layer
└── README.md                         # Project overview and instructions
```
