import dataclasses
import math
from typing import Optional

import rich
import torch


@dataclasses.dataclass
class VOD:
    log_p: torch.Tensor
    theta_loss: torch.Tensor
    phi_loss: Optional[torch.Tensor] = None

    @property
    def loss(self) -> torch.Tensor:
        if self.phi_loss is not None:
            return self.theta_loss + self.phi_loss
        return self.theta_loss


def vod_objective(
    log_p_x__z: torch.Tensor,
    log_p_z: torch.Tensor,
    log_q_z: torch.Tensor,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VOD:
    """Compute the VOD objective and its gradients.

    Args:
        log_p_x__z: log p(x|z), parameter `theta`
        log_p_z: log p(z), parameter `theta`
        log_q_z: log q(z|x) (sampling distribution, parameter `phi`)
        log_s: log s(z|x) (importance weights for the samples `z ~ q(z|x)`)
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

    # rich.print(dict(
    #     log_s=log_s[0],
    #     log_s_=log_s_[0],
    #     self_normalized_bound=self_normalized_bound[0],
    #     self_normalized_bound_=self_normalized_bound_[0],
    #     log_p_x__z=log_p_x__z[0],
    #     log_p_x__z_=log_p_x__z_[0],
    #     controlled_score=controlled_score[0],
    #     log_q_z=log_q_z[0],
    #     theta_loss=theta_loss.mean(),
    #     phi_loss=phi_loss.mean(),
    # ))

    return VOD(log_p=log_p, theta_loss=theta_loss, phi_loss=phi_loss)


def _sn_riw_bound(*, log_p_x__z: torch.Tensor, log_zeta: torch.Tensor, log_s: torch.Tensor, alpha: float = 0.0):
    """Compute the self-normalized Rényi importance weighted bound."""
    if alpha == 1.0:
        raise NotImplementedError("alpha=1.0 is not implemented yet (case RVB=ELBO).")
    log_s_normed = torch.log_softmax(log_s, dim=-1)
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s_normed + log_zeta, dim=-1, keepdim=True)
    self_normalized_bound = 1 / (1 - alpha) * torch.logsumexp(log_s_normed + (1 - alpha) * log_v, dim=-1)
    return self_normalized_bound
