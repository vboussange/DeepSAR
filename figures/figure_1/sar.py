# plotting expected SAR and dSAR, with sampling area dependent variation due to climate
# TOFIX: problem with the derivative plot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class ScalarFunctionModel(nn.Module):

    def forward(self, x):
        # return  1 - 0.5 * torch.exp(-x) - 0.5 * torch.exp(-x / 1000)
        # return  1 - 0.5 * torch.exp(-x / 1000)
        return 1 / (1 + torch.exp(-x)) + 2 / (1 + torch.exp(-(x - 1000) / 1000))

def plot_model_output_and_derivative(model, x, y, y_derivative, std):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Plot the output value with synthetic standard deviation
    ax = axes[0]
    ax.plot(x.detach().numpy(), y.detach().numpy(), label='Model Output')
    ax.fill_between(x.detach().numpy().flatten(), 
                    (y - std).detach().numpy().flatten(), 
                    (y + std).detach().numpy().flatten(), 
                    color='blue', alpha=0.2, label='Synthetic Std Dev')
    ax.set_xlabel('Area')
    ax.set_ylabel('Species richness')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    # Plot the derivative with synthetic standard deviation
    ax = axes[1]
    ax.plot(x.detach().numpy(), y_derivative.detach().numpy(), label='Model Derivative', color='r')
    # ax.fill_between(x.detach().numpy().flatten(), 
    #                 (y_derivative - std).detach().numpy().flatten(), 
    #                 (y_derivative + std).detach().numpy().flatten(), 
    #                 color='red', alpha=0.2, label='Synthetic Std Dev')
    ax.set_xlabel('x')
    ax.set_ylabel('d log Species richness / d log Area')
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_title('Model Derivative vs x')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Example usage
model = ScalarFunctionModel()

# Generate data
steps=100
# x = torch.linspace(0, 100, steps).unsqueeze(1)
x = torch.logspace(-2, 4, steps, base=10).unsqueeze(1)
y = model(x)

# Compute the derivative
x.requires_grad_(True)
y_derivative = torch.autograd.grad(outputs=model(x).sum(), inputs=x)[0]

# Create synthetic standard deviation that decreases with increasing x
std = 1e-1 * torch.exp(0.2 / (x * 1e-2 + 1))

# Plot the model output and its derivative
plot_model_output_and_derivative(model, x, y, y_derivative, std)