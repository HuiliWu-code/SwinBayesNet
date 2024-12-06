
# SwinBayesNet

SwinBayesNet is a deep learning method for searching for Hot Subdwarf Stars from SDSS images, combining Swin Transformer and Bayesian Neural Networks. The model is initially trained for two-class and three-class classification using the Swin Transformer. Once the model is trained, we freeze the backbone (feature extraction) and replace the output head with a Bayesian neural network to produce the final classification results and estimate uncertainty. For more details, refer to the paper.
![image](/SwinBayesNet.png)

## Contents

- **Backbone_Feature Extraction**: `Swin_transformer_model.py` contains the feature extraction model for SDSS images with 5 bands. We provide two models: one for two-class classification and one for three-class classification for the two-stage model mentioned in the paper. You can use the following code as an example:

    ```python
    # Swin Transformer model for two-class classification
    model = swin_two_classifiaction(num_classes=2)
    input = torch.ones((1, 5, 48, 48))
    output = model(input)
    print(f"Output shape for two-class classification: {output.shape}")
    ```

    ```python
    # Swin Transformer model for three-class classification
    model = swin_three_classifiaction(num_classes=3)
    input = torch.ones((1, 5, 48, 48))
    output = model(input)
    print(f"Output shape for three-class classification: {output.shape}")
    ```

- **Head_Classification + Uncertainty**: In the `Bayesian_model` folder, `Bayesian_head.py` contains the head for SwinBayesNet. It provides two models for two-class classification (1024 features) and three-class classification (1536 features). You can use the following code as an example:

    ```python
    # Bayesian model for two-class classification
    extracted_features = torch.ones((1, 1024)).to(device)
    bayesian_model = BayesianHead(input_features=1024, num_classes=2, priors=priors).to(device)
    output, kl_loss = bayesian_model(extracted_features)   # Output and KL loss
    print(f"Output for two-class classification: {output}")
    print(f"KL Loss: {kl_loss}")
    ```

    ```python
    # Bayesian model for three-class classification
    extracted_features = torch.ones((1, 1536)).to(device)
    bayesian_model = BayesianHead(input_features=1536, num_classes=3, priors=priors).to(device)
    output, kl_loss = bayesian_model(extracted_features)   # Output and KL loss
    print(f"Output for three-class classification: {output}")
    print(f"KL Loss: {kl_loss}")
    ```

## File Structure

```
├── Bayesian_model
│   ├── __init__.py                # Initialization file for the Bayesian model
│   ├── metrics.py                 # Metrics related to the model evaluation
│   ├── misc.py                    # Utility functions and helpers
│   ├── BBB_LRT_BBBLinear.py       # Bayesian Linear Layer (LRT)
│   ├── Bayesian_head.py           # Bayesian head for SwinBayesNet (classification + uncertainty)
├── Swin_transformer_model.py      # Swin Transformer backbone for feature extraction
└── README.md                      # Project overview and instructions
```
