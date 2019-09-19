from numpy.polynomial.hermite import hermgauss
import torch
import math


root_two = math.sqrt(2.0)


def quad_prob(mu, var, labels, K=2, NQ=10):
    """
    mu and var of shape B x N_classes.
    labels is of shape B.
    NQ is the number of quadrature points along each dimension.
    K is the number of dimensions to do quadrature along at cost NQ per dimension.
    """
    X, WX = hermgauss(NQ)
    X, WX = torch.tensor(X).type_as(mu), torch.tensor(WX).type_as(mu)

    N_classes = mu.size(-1)
    sigma = var.sqrt()

    ucb = mu + sigma
    ucb[torch.arange(mu.size(0)), labels] += 99999.9

    _, top_ucb = torch.topk(ucb, K, -1)
    _, bot_ucb = torch.topk(ucb, N_classes - K, -1, largest=False)
    top_sigma = torch.gather(sigma, -1, top_ucb)
    top_mu = torch.gather(mu, -1, top_ucb)
    bot_mu = torch.gather(mu, -1, bot_ucb)

    top_terms = []
    for dim in range(K):
        Xf = root_two * top_sigma[:, dim].unsqueeze(-1) * X + top_mu[:, dim].unsqueeze(-1)
        for d in range(dim):
            Xf = Xf.unsqueeze(-2)
        for d in range(K - dim - 1):
            Xf = Xf.unsqueeze(-1)
        top_terms.append(Xf.exp())

    bot_terms = sum([bot_mu[:, k].exp() for k in range(bot_mu.size(-1))])
    for dim in range(K):
        bot_terms = bot_terms.unsqueeze(-1)

    ratio = top_terms[0] / (sum(top_terms) + bot_terms)
    for dim in range(K):
        ratio *= WX
        WX = WX.unsqueeze(-1)

    for dim in range(K):
        ratio = ratio.sum(-1)

    prob = ratio / math.pow(math.pi, 0.5 * K)

    return prob
