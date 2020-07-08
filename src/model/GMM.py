"""
GMM: Gaussian Mixture Model

Representation of Gaussian Mixture Model probability distribution.
These functions allow to sample from and maximum-likelihood estimation of the parameters of a GMM
distribution.

Reference:
    * https://github.com/CPJKU/madmom/blob/master/madmom/ml/gmm.py
    * https://gist.github.com/lirnli/a10629fd068384d114c301db18e729c8
    * https://www.cs.bgu.ac.il/~cv201/wiki.files/logsumexp_trick.pdf
"""
import torch
from torch import Tensor


def constraint_mu(mu: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (3).

    Args:
        mu (Tensor): tensor representing the mean of the distribution. Shape: [i*j, K]

    Returns:
        mu_constrained (Tensor): Shape: [i*j, K]
    """
    return mu


def constraint_std(std: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (4).

    Args:
        std (Tensor): tensor representing the standard deviation of the distribution. Shape: [i*j, K]:

    Returns:
        std_constrained (Tensor): Shape: [i*j, K]
    """
    return torch.exp(std)


def constraint_pi(pi: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (5).

    Args:
        pi (Tensor): tensor representing the mixture coefficients of the distribution. Shape: [i*j, K]

    Returns:
        pi_constrained (Tensor): Shape: [i*j, K]
    """
    return torch.softmax(pi, 1)  # softmax accross columns


def sample_gmm(mu: Tensor, std: Tensor, pi: Tensor) -> Tensor:
    r"""
    Sample from i*j GMM distributions defined by \mu, \std and \pi, one GMM distribution for every x_{ij}.

    .. Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Args:
        mu (Tensor): means of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]
        std (Tensor): std of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]
        pi (Tensor): pi of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]

    Returns:
        samples (Tensor): One sample for every GMM corresponding to every x_{ij}. Shape: [i*j, 1]
    """
    # enforce constraints on parameters mu, std, pi according to (3), (4), (5)
    mu = mu.reshape(-1, mu.shape[-1])  # transform mu from [i, j, K] or [i*j, K] to [i*j, K]
    mu = constraint_mu(mu)
    std = std.reshape(-1, std.shape[-1])  # transform std from [i, j, K] or [i*j, K] to [i*j, K]
    std = constraint_std(std)
    pi = pi.reshape(-1, pi.shape[-1])  # transform pi from [i, j, K] or [i*j, K] to [i*j, K]
    pi = constraint_pi(pi)

    # choose component to sample for every x
    k = torch.multinomial(pi, num_samples=1)

    # choose mu and std according to the components for every x
    mu = mu.gather(1, k)
    std = std.gather(1, k)

    # sample from normal distribution from mu and std according to components for every x
    generated = torch.distributions.normal.Normal(mu, std).sample()
    return generated


def loss_gmm(mu: Tensor, std: Tensor, pi: Tensor, data: Tensor) -> Tensor:
    r"""
    Loss from i*j GMM distributions defined by \mu, \std and \pi, one GMM distribution for every x_{ij},
    with respect to \data.

    .. Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Args:
        mu (Tensor): means of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]
        std (Tensor): std of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]
        pi (Tensor): pi of GMM with k components for every x_{ij}. Shape: [i, j, K] or [i*j, K]
        data (Tensor): real data to calculate the loss. Shape: shape [i, j] or [i*j]

    Returns:
        total_loss: sum of the individual loss of every x_{ij}, parameterized by theta_{ij} GMM
                    distribution, with respect to data_{ij}. Shape: [] (0 dimension)
    """
    # enforce constraints on parameters mu, std, pi according to (3), (4), (5)
    mu = mu.reshape(-1, mu.shape[-1])  # transform mu from [i, j, K] or [i*j, K] to [i*j, K]
    mu = constraint_mu(mu)
    std = std.reshape(-1, std.shape[-1])  # transform std from [i, j, K] or [i*j, K] to [i*j, K]
    std = constraint_std(std)
    pi = pi.reshape(-1, pi.shape[-1])  # transform pi from [i, j, K] or [i*j, K] to [i*j, K]
    pi = constraint_pi(pi)

    # modify data to fit the number of components of the GMM
    data = data.reshape(-1, 1)  # transform data from [i, j] or [i*j] to [i*j, 1]
    data = data.expand(mu.shape)  # transform data from [i*j, 1] to [i*j, K] duplicating columns

    # use Log Sum Exp trick to calculate the loss (negative log-likelihood)
    log_pi = torch.log(pi)
    log_pdf = torch.distributions.normal.Normal(mu, std).log_prob(data)

    # this vector [i*j, 1] represents the individual loss of every pixel with respect to data
    loss = -torch.logsumexp(log_pi + log_pdf, dim=1, keepdim=True)

    # return the total loss of all the pixels
    return loss.sum()
