import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

train_x = train_x.unsqueeze(-1).cuda()
train_y = train_y.cuda()

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        devices = [0, 1]
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(base_covar_module, device_ids=devices, output_device=torch.device('cuda', 0))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to("cuda:0")
model = ExactGPModel(train_x, train_y, likelihood).to("cuda:0")

foo = model.covar_module(train_x).evaluate_kernel()
print(foo.shape, foo.device)
print(type(foo.lazy_tensors[0]), foo.lazy_tensors[0].device)
print(foo.lazy_tensors[1].device)
bar = foo.lazy_tensors[0].evaluate_kernel()
print(bar.device, bar.shape)
baz = foo.lazy_tensors[1].evaluate_kernel()
print(baz.device, baz.shape)
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

with gpytorch.settings.max_preconditioner_size(5):
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.module.base_kernel.log_lengthscale.item(),
            model.likelihood.log_noise.item()
        ))
        optimizer.step()

model.eval()
with torch.no_grad():
    observed_pred = model(train_x)

print((observed_pred.mean - train_y).mean().abs())

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(train_x.cpu().numpy(), observed_pred.mean.cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(train_x.squeeze().cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.savefig('res.pdf')
