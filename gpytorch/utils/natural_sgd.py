from torch.optim import SGD
from torch.optim.optimizer import required
import torch


class NaturalSGD(SGD):
    def __init__(
        self,
        nat_params,
        other_params,
        variational_dist,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        from ..variational.natural_variational_distribution import NaturalVariationalDistribution
        if not isinstance(variational_dist, NaturalVariationalDistribution):
            raise RuntimeError("NaturalSGD can only be used with a NaturalVariationalDistribution!")

        # nat_params are the natural parameters we want to precondition with the Fisher matrix
        self.nat_params = nat_params

        # Other parameters are all other parameters, to which we'll apply normal Adam
        self.other_params = other_params
        self.variational_dist = variational_dist
        params = list(nat_params) + list(other_params)
        super(NaturalSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, loss):
        # Get buffered mu, L
        mu, L = self.variational_dist.buffer
        etas = self.variational_dist._meanvarsqrt_to_expectation(mu, L)

        # Compute gradients w.r.t mu and L
        standard_grads = torch.autograd.grad(loss, [mu, L], retain_graph=True)

        # Compute gradients w.r.t. eta1 and eta2
        nat_grads = torch.autograd.grad(
            self.variational_dist._expectation_to_meanvarsqrt(*etas),
            etas,
            grad_outputs=standard_grads,
            retain_graph=True
        )

        # Get all other gradients
        loss.backward()

        # Overwrite variational parameter gradients
        self.variational_dist.natural_variational_mean.grad = nat_grads[0].contiguous()
        self.variational_dist.natural_variational_covar.grad = nat_grads[1].contiguous()

        # Take a standard Adam step
        super(NaturalSGD, self).step()

        # Mark variational distribution buffer as dirty
        self.variational_dist.has_buffer = False
