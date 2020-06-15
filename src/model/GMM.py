"""
GMM: Gaussian Mixture Model

Representation of Gaussian Mixture Model probability distribution.
These functions allow to sample from and maximum-likelihood estimation of the parameters of a GMM
distribution.

References:
    - https://github.com/CPJKU/madmom/blob/master/madmom/ml/gmm.py
    - https://gist.github.com/lirnli/a10629fd068384d114c301db18e729c8
    - https://www.cs.bgu.ac.il/~cv201/wiki.files/logsumexp_trick.pdf
"""
import torch
from torch import Tensor


def constraint_mu(mu: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (3).

    Args:
        mu (Tensor), shape (i*j, K): tensor representing the mean of the distribution.
    """
    return mu


def constraint_std(std: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (4).

    Args:
        std (Tensor), shape (i*j, K): tensor representing the standard deviation of the distribution.
    """
    return torch.exp(std)


def constraint_pi(pi: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (5).

    Args:
        pi (Tensor), shape (i*j, K): tensor representing the mixture coefficients of the distribution.
    """
    return torch.softmax(pi, 1) # softmax accross columns


def sample_gmm(mu: Tensor, std: Tensor, pi: Tensor) -> Tensor:
    r"""
    Sample from i*j GMM distributions defined by \mu, \std and \pi, one GMM distribution for every x.

    Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Arguments:
        mu (Tensor), shape (i, j, K) or (i*j, K): means of GMM with k components for every x
        std (Tensor), shape (i, j, K) or (i*j, K): std of GMM with k components for every x
        pi (Tensor), shape (i, j, K) or (i*j, K): pi of GMM with k components for every x

    Returns:
        shape (i*j, 1): One sample for every GMM corresponding to every x
    """
    # enforce constraints on parameters mu, std, pi according to (3), (4), (5)
    mu = mu.reshape(-1, mu.shape[-1])
    mu = constraint_mu(mu)
    std = std.reshape(-1, std.shape[-1])
    std = constraint_std(std)
    pi = pi.reshape(-1, pi.shape[-1])
    pi = constraint_pi(pi)

    # choose component to sample for every x
    k = torch.multinomial(pi, num_samples=1)

    # choose mu and std according to the components for every x
    mu = mu.gather(1, k)
    std = std.gather(1, k)

    # sample from normal distribution from mu and std according to components for every x
    generated = torch.distributions.normal.Normal(mu, std).sample()
    return generated


def loss_gmm(mu: Tensor, std: Tensor, pi: Tensor, x: Tensor) -> Tensor:
    r"""
    Loss from i*j GMM distributions defined by \mu, \std and \pi, one GMM distribution for every x
    with respect to \data.

    Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Arguments:
        mu (Tensor), shape (i, j, K) or (i*j, K): means of GMM with k components for every x
        std (Tensor), shape (i, j, K) or (i*j, K): std of GMM with k components for every x
        pi (Tensor), shape (i, j, K) or (i*j, K): pi of GMM with k components for every x
        x (Tensor), shape (i, j) or (i*j): real data to calculate the loss

    Returns:
        shape (1): Loss of every x_{ij} with respect to the theta_{ij} GMM distribution
    """
    # enforce constraints on parameters mu, std, pi according to (3), (4), (5)
    mu = mu.reshape(-1, mu.shape[-1])
    mu = constraint_mu(mu)
    std = std.reshape(-1, std.shape[-1])
    std = constraint_std(std)
    pi = pi.reshape(-1, pi.shape[-1])
    pi = constraint_pi(pi)

    # modify data to fit the number of components of the GMM
    x = x.expand(mu.size())

    # use Log Sum Exp trick to calculate the loss
    log_pi = torch.log(pi)
    log_pdf = torch.distributions.normal.Normal(mu, std).log_prob(x)
    loss = torch.logsumexp(log_pi + log_pdf, dim=1, keepdim=True)
    return loss.mean()
