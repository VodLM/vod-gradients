from __future__ import annotations

import typing

import torch
from torch import distributions as torch_dist
from torch import nn

import vod


class GaussianMixtureModel(nn.Module):
    r"""A Gaussian Mixture Model.
    
    A simple VAE model parametrized by MLPs as described in
    `Revisiting Reweighted Wake-Sleep for Models with Stochastic Control Flow` [https://arxiv.org/abs/1805.10469].

    * p_{\theta}(z) = Cat(z | softmax(\theta)), p(x | z)= \mathcal{N} (x | \mu_{z}, \sigma_{z}^{2} ) \\
    * q_{\phi}(z | x) = Cat( z | softmax(\eta_{\phi}(x) )
    * \eta_{\phi}(x) : 1-16-C
    """

    def __init__(
        self,
        n: int = 20,
        hdim: int = 16,
        p_scale: float = 5.0,
        p_mu_increment: float = 10.0,
        **kwargs,
    ) -> None:
        """Initialise a Gaussian-Mixture model.

        :param N: number of clusters
        :param hdim: hidden dimensions of the perceptrons
        :param kwargs:
        """
        super().__init__()
        xdim = 1
        self.C = n
        self.register_buffer("log_theta_opt", torch.log(5 + torch.arange(0, self.C, dtype=torch.float)).view(1, self.C))
        self.log_theta = nn.Parameter(torch.zeros(1, self.C))
        self.register_buffer("p_mu", p_mu_increment * torch.arange(0, self.C)[:, None])
        self.register_buffer("p_scale", torch.tensor(p_scale))
        self.phi = nn.Sequential(nn.Linear(xdim, hdim), nn.Tanh(), nn.Linear(hdim, self.C))
        self.phi[-1].weight.data.zero_()

    def approx_posterior(self, x: torch.Tensor) -> torch_dist.Categorical:
        logits = self.phi(x)
        return torch_dist.Categorical(logits=logits)

    def true_posterior(self, x: torch.Tensor) -> torch_dist.Categorical:
        M, C = x.size(0), self.C
        x = x.view(M, 1, 1).expand(M, C, 1)
        z = torch.arange(C, device=x.device)[None, :]
        p_x__z = self.observation_model(z)
        log_p_x_z = p_x__z.log_prob(x)
        log_posterior = self.log_theta_opt.sum(1) + log_p_x_z.sum(-1)
        return torch_dist.Categorical(logits=log_posterior.log_softmax(dim=-1))

    def observation_model(self, z: torch.Tensor) -> torch_dist.Normal:
        mu = self.p_mu[z]
        return torch_dist.Normal(loc=mu, scale=self.p_scale)

    def log_prob(self, x: torch.Tensor, from_optimal: bool = False) -> torch.Tensor:
        zs = torch.arange(self.C, device=x.device)[None, :].expand(x.size(0), self.C)
        log_p_x__z = self.observation_model(zs)
        log_p_x_z = log_p_x__z.log_prob(x[:, None, :]).sum(-1)
        prior_ = self.log_theta_opt if from_optimal else self.log_theta
        return torch.logsumexp(prior_[None, :].log_softmax(dim=-1) + log_p_x_z, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        n_samples: int = 8,
        sampler: typing.Literal["topk", "multinomial", "priority"] = "multinomial",
        sample_prior: bool = False,
        **kwargs: typing.Any,  # noqa: ANN
    ) -> dict[str, torch.Tensor]:
        qz = self.approx_posterior(x)
        pz = torch_dist.Categorical(logits=self.log_theta.expand(x.size(0), self.C))
        sample_dist = {True: qz, False: pz}[sample_prior]
        sample_fn = {
            "multinomial": vod.multinomial_sample,
            "priority": vod.priority_sample,
            "topk": vod.topk_sample,
        }[sampler]
        samples = sample_fn(sample_dist.logits, n_samples)
        px = self.observation_model(samples.indices)
        diagnostics = self._get_diagnostics(x, qz)
        return {
            "px": px,
            "z": samples.indices,
            "log_s": samples.log_weights,
            "qz": qz,
            "pz": pz,
            **diagnostics,
        }

    def sample_from_prior(self, n: int, from_optimal=False, **kwargs):
        prior = self.log_theta_opt.expand(n, self.C) if from_optimal else self.log_theta.expand(n, self.C)
        z = torch_dist.Categorical(logits=prior).sample()
        px = self.observation_model(z)
        return {"px": px, "z": z}

    @torch.no_grad()
    def _get_diagnostics(self, x: torch.Tensor, qz: torch_dist.Categorical):
        return {
            "prior_kl": _kldiv(self.log_theta, self.log_theta_opt).mean(),
            "posterior_kl": _kldiv(qz.logits, self.true_posterior(x).logits).mean(),
            "logp_true": self.log_prob(x, from_optimal=True).mean(),
            "logp_model": self.log_prob(x, from_optimal=False).mean(),
        }


def _kldiv(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    log_q = q.log_softmax(dim=-1)
    log_p = p.log_softmax(dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(-1)
