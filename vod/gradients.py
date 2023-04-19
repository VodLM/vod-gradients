import dataclasses
import math
from typing import Optional

import torch


@dataclasses.dataclass
class VOD:
    log_p: torch.Tensor
    elbo: torch.Tensor
    theta_loss: torch.Tensor
    phi_loss: Optional[torch.Tensor] = None

    @property
    def loss(self) -> torch.Tensor:
        if self.phi_loss is not None:
            return self.theta_loss + self.phi_loss
        return self.theta_loss

    @property
    def kl(self) -> torch.Tensor:
        return self.log_p - self.elbo


    def dict(self) -> dict[str, torch.Tensor]:
        return {
            "log_p": self.log_p,
            "elbo": self.elbo,
            "theta_loss": self.theta_loss,
            "phi_loss": self.phi_loss,
            "loss": self.loss,
            "kl": self.kl,
        }


def vod_objective(
    log_p_x__z: torch.Tensor,
    log_p_z: torch.Tensor,
    log_q_z: torch.Tensor,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VOD:
    """Compute the VOD objective and its gradients.

    Args:
        log_p_x__z: log p(x|z), parameter `theta` (unnormalized)
        log_p_z: log p(z), parameter `theta` (unnormalized)
        log_q_z: log q(z|x) (sampling distribution, parameter `phi`) (unnormalized)
        log_s: log s(z|x) (importance weights for the samples `z ~ q(z|x)`)s
        alpha: Rényi alpha parameter

    Returns:
        VOD: VOD bound and its gradients
    """
    bs, n_samples = log_p_x__z.shape
    if log_s is None:
        log_s = torch.zeros_like(log_q_z)

    log_zeta = log_p_z - log_q_z

    # compute the IWB
    log_p = _sn_riw_bound(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s,
        alpha=0,
    )

    # compute the ELBO
    elbo = _sn_riw_bound(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s,
        alpha=1,
    )

    # compute the approximate IW-RVB (VOD)
    self_normalized_bound = _sn_riw_bound(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s, alpha=alpha)

    # compute the gradients for `log p(z)`
    log_p_z_grad = log_p_z - torch.sum(torch.softmax(log_s + log_zeta, dim=-1).detach() * log_p_z, dim=-1, keepdim=True)

    # compute the loss for `theta`
    w = torch.softmax(log_s + (1 - alpha) * (log_p_x__z + log_zeta), dim=-1)
    theta_loss = -torch.sum(w.detach() * (log_p_x__z + log_p_z_grad), dim=-1)

    # compute the loss for `phi`
    with torch.no_grad():
        log_p_x__z_ = log_p_x__z[:, None, :].expand(bs, n_samples, n_samples)
        log_zeta_ = log_zeta[:, None, :].expand(bs, n_samples, n_samples)
        log_s_ = log_s[:, None, :].expand(bs, n_samples, n_samples)
        ids = torch.arange(n_samples, device=log_p_x__z.device)[None, :, None]
        log_p_x__z_ = log_p_x__z_.scatter(-1, ids, -math.inf)
        log_zeta_ = log_zeta_.scatter(-1, ids, -math.inf)
        log_s_ = log_s_.scatter(-1, ids, -math.inf)
        self_normalized_bound_ = _sn_riw_bound(log_p_x__z=log_p_x__z_, log_zeta=log_zeta_, log_s=log_s_, alpha=alpha)
        controlled_score = self_normalized_bound[:, None] - self_normalized_bound_
    phi_loss = -torch.sum(controlled_score.detach() * log_q_z, dim=-1)

    return VOD(log_p=log_p, elbo=elbo, theta_loss=theta_loss, phi_loss=phi_loss)


def _elbo(*, log_p_x__z: torch.Tensor, log_zeta: torch.Tensor, log_s: torch.Tensor):
    """Compute the ELBO."""
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s + log_zeta, dim=-1, keepdim=True)
    log_v_ = torch.where(torch.isinf(log_v), torch.zeros_like(log_v), log_v)
    elbo = torch.sum(log_s.exp() * log_v_, dim=-1)
    return elbo


def _sn_riw_bound(*, log_p_x__z: torch.Tensor, log_zeta: torch.Tensor, log_s: torch.Tensor, alpha: float = 0.0):
    """Compute the self-normalized Rényi importance weighted bound."""
    log_s_norm = log_s.log_softmax(dim=-1)
    if alpha == 1.0:
        return _elbo(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s_norm)
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s_norm + log_zeta, dim=-1, keepdim=True)
    # avoid numerical issues: in the case $\alpha > 1$, we need to reverse the sign of the masked `log_v` (`inf`).
    log_v = torch.where(log_v.isinf() & (log_v > 0), -log_v, log_v)
    self_normalized_bound = 1 / (1 - alpha) * torch.logsumexp(log_s_norm + (1 - alpha) * log_v, dim=-1)
    return self_normalized_bound
