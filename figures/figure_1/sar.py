# plotting expected SAR and dSAR, with sampling area dependent variation due to climate
# TOFIX: problem with the derivative plot with autodiff, using finite difference for now
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams.update({'axes.labelsize': 14})

class ScalarFunctionModel(nn.Module):

    def forward(self, x):
        # return  1 - 0.5 * torch.exp(-x) - 0.5 * torch.exp(-x / 1000)
        # return  1 - 0.5 * torch.exp(-x / 1000)
        return torch.log(1 / (1 + torch.exp(-x)) + 2 / (1 + torch.exp(-(x - 1000) / 1000)))

def plot_model_output_and_derivative(model, x, y, y_derivative, std):
    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)

    # Plot the output value with synthetic standard deviation
    ax = axes[0]
    ax.plot(x.detach().numpy(), y.detach().numpy())
    ax.fill_between(x.detach().numpy().flatten(), 
                    (y * torch.exp(std)).detach().numpy().flatten(), 
                    (y / torch.exp(std)).detach().numpy().flatten(), 
                    color='blue', alpha=0.2)
    ax.set_ylabel('Species richness (SR)')
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Plot the derivative with synthetic standard deviation
    ax = axes[1]
    ax.plot(x.detach().numpy(), y_derivative.detach().numpy().flatten(), color='r')
    ax.fill_between(x.detach().numpy().flatten(), 
                    (y_derivative + 1e-1 * std).detach().numpy().flatten(), 
                    (y_derivative - 1e-1 * std).detach().numpy().flatten(), 
                    color='red', alpha=0.2)
    ax.set_ylabel(r"$\frac{d \log(SR)}{d \log(A)}$", fontsize=18)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_title('Model Derivative vs x')
    ax.set_xlabel('Area')

    fig.tight_layout()
    return fig, axes

model = ScalarFunctionModel()
steps=100

# forward pass calculation
x = torch.logspace(-2, 4, steps, base=10).unsqueeze(1)
x.requires_grad_(True)
y = model(x)

# backward pass calculation, autodiff - not working
# x = torch.linspace(-2, 4, steps).unsqueeze(1)
# x = torch.pow(10, x)
# y_derivative = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))[0]

# backward pass calculation, finite difference
dx = x[1] - x[0]
y_finite_diff = (y[2:] - y[:-2]) / (2 * dx) / 30
x_finite_diff = x[1:-1]
y = y[1:-1]
x = x[1:-1]
# plt.plot(x, y_finite_diff)
# plt.xscale("log")

# Synthetic standard deviation
std = 0.2 * (-torch.log((x-x.min()/(x.max() - x.min()))) + 9 )

# Plot the model output and its derivative
fig, axs = plot_model_output_and_derivative(model, x, torch.exp(1e1*y), y_finite_diff, std)
axs[0].set_ylim(1, axs[0].get_ylim()[1])
axs[1].set_ylim(0, axs[1].get_ylim()[1])

for ax in axs:
    ax.set_xlim(5e-2, ax.get_xlim()[1])
    # ax.tick_params(axis='both', which='both', length=0)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    
fig.savefig("conceptual_SAR.pdf", bbox_inches='tight', dpi=300)