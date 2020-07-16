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
import torch.nn as nn


def constraint_mu(mu: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (3).

    Args:
        mu (Tensor): tensor representing the mean of the distribution.
                     Shape: [..., K] (e.g. [i*j, K] or [B, i, j, K])

    Returns:
        mu_constrained (Tensor): Shape: original shape [..., K] (e.g. [i*j, K] or [B, i, j, K])
    """
    return mu


def constraint_std(std: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (4).

    Args:
        std (Tensor): tensor representing the standard deviation of the distribution.
                      Shape: [..., K] (e.g. [i*j, K] or [B, i, j, K])

    Returns:
        std_constrained (Tensor): Shape: original shape [..., K] (e.g. [i*j, K] or [B, i, j, K])
    """
    return torch.exp(std)


def constraint_pi(pi: Tensor) -> Tensor:
    r"""
    Constraint according to MelNet formula (5).

    Args:
        pi (Tensor): tensor representing the mixture coefficients of the distribution.
                     Shape: [..., K] (e.g. [i*j, K] or [B, i, j, K])

    Returns:
        pi_constrained (Tensor): Shape: original shape [..., K] (e.g. [i*j, K] or [B, i, j, K])
    """
    return torch.softmax(pi, dim=-1)  # softmax accross last dimension (K components)


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
    # enforce constraints on parameters mu, std, pi according to MelNet formula (3), (4), (5)
    mu = mu.reshape(-1, mu.shape[-1])  # transform mu from [i, j, K] or [i*j, K] to [i*j, K]
    mu = constraint_mu(mu)
    std = std.reshape(-1, std.shape[-1])  # transform std from [i, j, K] or [i*j, K] to [i*j, K]
    std = constraint_std(std)
    pi = pi.reshape(-1, pi.shape[-1])  # transform pi from [i, j, K] or [i*j, K] to [i*j, K]
    pi = constraint_pi(pi)

    # choose component to sample for every x
    k_idx = torch.distributions.categorical.Categorical(probs=pi).sample().unsqueeze(dim=-1)

    # choose mu and std according to the components for every x
    mu = mu.gather(dim=1, index=k_idx)
    std = std.gather(dim=1, index=k_idx)

    # sample from normal distribution from mu and std according to components for every x
    generated = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
    return generated


def sample_gmm_batch(mu_hat: Tensor, std_hat: Tensor, pi_hat: Tensor) -> Tensor:
    r"""
    Sample B times (where B is batch size) from i*j GMM distributions defined by \mu, \std and
    \pi, one GMM distribution for every x_{ij}.

    .. Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Args:
        mu_hat (Tensor): unconstrained means of GMM with k components for every x_{ij} for every
                         item in batch of size B. Shape: [B, i, j, K]
        std_hat (Tensor): unconstrained std of GMM with k components for every x_{ij} for every
                          item in batch of size B. Shape: [B, i, j, K]
        pi_hat (Tensor): unconstrained pi of GMM with k components for every x_{ij} for every item
                         in batch of size B. Shape: [B, i, j, K]

    Returns:
        samples (Tensor): One sample for every GMM corresponding to every x_{ij} for every item
                          in batch of size B. Shape: [B, i, j]
    """
    # enforce constraints on parameters mu, std, pi according to MelNet formula (3), (4), (5)
    mu = constraint_mu(mu_hat)
    std = constraint_std(std_hat)
    pi = constraint_pi(pi_hat)

    # choose component to sample for every x_ij in an item in the batch
    k_idx = torch.distributions.categorical.Categorical(probs=pi).sample().unsqueeze(dim=-1)

    # choose mu and std according to the components for every x
    # it changes shape from [B, i, j, K] to [B, i, j, 1] because for every x_ij in an item in
    # the batch it chooses one out of K components to sample from
    mu = mu.gather(dim=3, index=k_idx)
    std = std.gather(dim=3, index=k_idx)

    # sample from normal distribution from mu and std according to components for every x in an item
    # in the batch
    generated = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
    # change shape from [B, i, j, 1] to [B, i, j]
    generated = generated.squeeze(dim=-1)
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
    # enforce constraints on parameters mu, std, pi according to MelNet formula (3), (4), (5)
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


def loss_gmm_batch(mu_hat: Tensor, std_hat: Tensor, pi_hat: Tensor, data: Tensor) -> Tensor:
    r"""
    Loss of B items, each with i*j GMM distributions defined by \mu, \std and \pi, one GMM
    distribution for every x_{ij}, with respect to \data.

    .. Note:
        theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
        modelled by a Gaussian Mixture Model with k components.

    Args:
        mu_hat (Tensor): unconstrained means of GMM with k components for every x_{ij} for every
                         item in batch of size B. Shape: [B, i, j, K]
        std_hat (Tensor): unconstrained std of GMM with k components for every x_{ij} for every
                          item in batch of size B. Shape: [B, i, j, K]
        pi_hat (Tensor): unconstrained pi of GMM with k components for every x_{ij} for every
                         item in batch of size B. Shape: [B, i, j, K]
        data (Tensor): real data to calculate the loss. Shape: shape [B, i, j]

    Returns:
        total_loss: sum of the individual loss of every item in batch of size B with respect to data.
                    Shape: [] (0 dimension)
        """
    # enforce constraints on parameters mu, std, pi according to MelNet formula (3), (4), (5)
    mu = constraint_mu(mu_hat)
    std = constraint_std(std_hat)
    # if std_hat was very negative, std would be very close to 0 and because there is a limited
    # precision, std would be exactly 0, causing a problem when calculating log-probabilities
    # For this reason, we clamp the values of std that are 0 to a value close to 0
    std = torch.clamp(std, min=0.00001)

    pi = constraint_pi(pi_hat)

    # modify data to fit the number of components of the GMM. For that:
    # - change shape of data from [B, i, j] to [B, i, j, 1]
    data = data.unsqueeze(dim=-1)
    # duplicate data for every component K. New shape is [B, i, j, K]
    data = data.expand(mu.shape)

    # use Log Sum Exp trick to calculate the loss (negative log-likelihood)
    log_pi = torch.log(pi)
    log_pdf = torch.distributions.normal.Normal(mu, std).log_prob(data)

    # this tensor [B, i,j , 1] represents the individual loss of the parameters for every pixel
    # (tetha_{ij} of every item in batch) with respect to data (data_{ij} of every item in batch)
    loss = -torch.logsumexp(log_pi + log_pdf, dim=-1, keepdim=True)

    # return the total loss of all the pixels of all items in batch
    return loss.sum()


class GMMLoss(nn.Module):
    r"""
    The negative log-likelihood loss for a Gaussian Mixture Model (GMM).

    Every item is parameterized by a Gaussian Mixture Model with K components:

    .. math::
        p(x_{ij}) = \sum_{k=1}^K \pi_{ijk} \mathcal{N}(x_{ijk}; \mu_{ijk}, \sigma_{ijk}

    and the negative log-likelihood loss can be described as:

    .. math::
        - log L(\theta_{ij} | x_{ij})
        = - log \sum_{k=1}^K \pi_{ijk} \mathcal{N}(x_{ijk}; \mu_{ijk}, \sigma_{ijk}

    which using the log-sum-exp trick is:

    .. math::
        - log \sum_{k=1}^K exp(log(\pi_{ijk} \mathcal{N}(x_{ijk}; \mu_{ijk}, \sigma_{ijk}))
        = - log \sum_{k=1}^K exp(log(\pi_{ijk})+log(\mathcal{N}(x_{ijk}; \mu_{ijk}, \sigma_{ijk}))

    To reduce the loss of all pixels of all items in the batch, the operation used is the sum.
    The general negative log-likelihood of one batch is:

    .. math::
        loss = \sum_{i=1}^I \sum_{j=1}^J - log L(\theta_{ij} | x_{ij})

    and the total loss across batches is just the sum of the loss of every batch.
    """

    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, mu: Tensor, std: Tensor, pi: Tensor, target: Tensor) -> Tensor:
        r"""
        Loss of B items, each with i*j GMM distributions defined by \mu, \std and \pi, one GMM
        distribution for every x_{ij}, with respect to \target_{ij}.

        .. Note:
            theta_{ij} = {mu_{ij}, std_{ij}, pi_{ij}} parameterizes a univariate density over x_{ij}
            modelled by a Gaussian Mixture Model with k components.

        Args:
            mu (Tensor): means of GMM with k components for every x_{ij} for every item in batch of
                         size B. Shape: [B, i, j, K]
            std (Tensor): std of GMM with k components for every x_{ij} for every item in batch of
                         size B. Shape: [B, i, j, K]
            pi (Tensor): pi of GMM with k components for every x_{ij} for every item in batch of
                         size B. Shape: [B, i, j, K]
            target (Tensor): real data to calculate the loss. Shape: shape [B, i, j]

        Returns:
            total_loss: sum of the individual loss of every item in batch of size B with respect to
                        target. Shape: [] (0 dimension)
        """
        return loss_gmm_batch(mu, std, pi, target)
