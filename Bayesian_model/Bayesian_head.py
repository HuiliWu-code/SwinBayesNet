
import torch
import torch.nn as nn
from misc import ModuleWrapper
from BBB_LRT_BBBLinear import BBB_LRT_BBBLinear


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesianHead(ModuleWrapper):

    """
    A Bayesian head module for handling features extracted from the feature extractor model
    and incorporating uncertainty estimation.
    """

    def __init__(self, input_features, hidden_features=256, middle_features=32, num_classes=2,
                 priors=None, act_layer=nn.GELU, drop=0.):
        super(BayesianHead, self).__init__()
        self.num_classes = num_classes

        # Initialize layers
        self.fc1 = BBB_LRT_BBBLinear(input_features, hidden_features, priors=priors)
        self.act = act_layer()
        self.fc2 = BBB_LRT_BBBLinear(hidden_features, middle_features, priors=priors)
        self.drop = nn.Dropout(drop)
        self.output = BBB_LRT_BBBLinear(middle_features, num_classes, priors=priors)  # Output layer

# Example usage
if __name__ == "__main__":
    priors = {
        'prior_mu': 0,
        'prior_sigma': 0.1,
        'posterior_mu_initial': (0, 0.1),
        'posterior_rho_initial': (-5, 0.1)
    }

    # Bayesian model for two-class classification
    extracted_features = torch.ones((1, 1024)).to(device)
    bayesian_model = BayesianHead(input_features=1024, num_classes=2, priors=priors).to(device)
    output, kl_loss = bayesian_model(extracted_features)   # Output and KL loss
    print(f"Output for two-class classification: {output.shape}")
    print(f"KL Loss: {kl_loss}")

    # Bayesian model for three-class classification
    extracted_features = torch.ones((1, 1536)).to(device)
    bayesian_model = BayesianHead(input_features=1536, num_classes=3, priors=priors).to(device)
    output, kl_loss = bayesian_model(extracted_features)   # Output and KL loss
    print(f"Output for three-class classification: {output.shape}")
    print(f"KL Loss: {kl_loss}")
