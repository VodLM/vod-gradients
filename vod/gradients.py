import dataclasses
import math
from typing import Optional

import rich
import torch


@dataclasses.dataclass
class VodObjective:
    logp: torch.Tensor
    elbo: torch.Tensor
    iw_rvb: torch.Tensor
    theta_loss: torch.Tensor
    phi_loss: Optional[torch.Tensor] = None

    @property
    def loss(self) -> torch.Tensor:
        if self.phi_loss is not None:
            return self.theta_loss + self.phi_loss
        return self.theta_loss

    @property
    def kl_vi(self) -> torch.Tensor:
        return self.logp - self.elbo

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "logp": self.logp,
            "elbo": self.elbo,
            "iw_rvb": self.iw_rvb,
            "theta_loss": self.theta_loss,
            "phi_loss": self.phi_loss,
            "kl_vi": self.kl_vi,
        }


def vod_objective(
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: torch.Tensor,
    log_s_psi: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VodObjective:
    """Compute the VOD objective and its gradients.

    Args:
        log_p_x__z: \propto log p(x|z), parameter `theta` (unnormalized)
        f_theta: \propt log p(z), parameter `theta` (unnormalized)
        f_phi: \propt log q(z|x) (approximate posterior, parameter `phi`) (unnormalized)
        f_psi: \propt log q(z|x) (sampling distribution, parameter `psi`) (unnormalized)
        log_s_psi: importance weights for the samples `z ~ q_\psi(z|x)`
        alpha: Rényi alpha parameter

    Returns:
        VOD: VOD bound and its gradients
    """
    if log_s_psi is None:
        log_s_psi = torch.zeros_like(f_psi)

    log_zeta = f_theta - f_psi

    # compute the IWB
    logp = _estimate_iw_rvb(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s_psi,
        alpha=0,
    )

    # compute the ELBO
    elbo = _estimate_iw_rvb(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s_psi,
        alpha=1,
    )

    # compute the approximate IW-RVB (VOD)
    iw_rvb = _estimate_iw_rvb(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s_psi, alpha=alpha)

    # compute the gradients for `\nabla_theta log p(z)`
    log_p_z_grad = f_theta - torch.sum(
        torch.softmax(log_s_psi + log_zeta, dim=-1).detach() * f_theta,
        dim=-1,
        keepdim=True,
    )

    # compute the loss for `theta`
    w = torch.softmax(log_s_psi + (1 - alpha) * (log_p_x__z + log_zeta), dim=-1)
    theta_loss = -torch.sum(w.detach() * (log_p_x__z + log_p_z_grad), dim=-1)

    # compute the loss for `phi`
    phi_loss = _vod_phi_loss(log_p_x__z, f_theta, f_phi, f_psi, log_s_psi, alpha=alpha)

    return VodObjective(logp=logp, elbo=elbo, iw_rvb=iw_rvb, theta_loss=theta_loss, phi_loss=phi_loss)


def _vod_phi_loss(
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: torch.Tensor,
    log_s_psi: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # rich.print(dict(
    #     log_p_x__z=log_p_x__z.shape,
    #     f_theta=f_theta.shape,
    #     f_phi=f_phi.shape,
    #     f_psi=f_psi.shape,
    #     log_s_psi=log_s_psi.shape,
    #     alpha=alpha,
    # ))

    bs, n_samples = log_p_x__z.shape
    log_zeta = f_theta - f_phi
    los_s_phi = log_s_psi - f_psi + f_phi
    s_phi = los_s_phi.softmax(dim=-1)
    with torch.no_grad():
        log_p_x__z_ = log_p_x__z[:, None, :].expand(bs, n_samples, n_samples)
        log_zeta_ = log_zeta[:, None, :].expand(bs, n_samples, n_samples)
        los_s_phi_ = los_s_phi[:, None, :].expand(bs, n_samples, n_samples)
        ids = torch.arange(n_samples, device=log_p_x__z.device)[None, :, None]
        log_p_x__z_ = log_p_x__z_.scatter(-1, ids, -math.inf)
        log_zeta_ = log_zeta_.scatter(-1, ids, -math.inf)
        los_s_phi_ = los_s_phi_.scatter(-1, ids, -math.inf)

        iw_rvb = _estimate_iw_rvb(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=los_s_phi, alpha=alpha)
        iw_rvb_ = _estimate_iw_rvb(log_p_x__z=log_p_x__z_, log_zeta=log_zeta_, log_s=los_s_phi_, alpha=alpha)
        controlled_score = iw_rvb[:, None] - iw_rvb_

    # compute the gradients for `\nabla_theta log p(z)`
    log_q_z_grad = f_phi - torch.sum(s_phi.detach() * f_phi, dim=-1, keepdim=True)
    # rich.print(dict(log_q_z_grad=log_q_z_grad[0], controlled_score=controlled_score[0], log_p_x__z=log_p_x__z[0], log_p_x__z_=log_p_x__z_[0], iw_rvb=iw_rvb[0], iw_rvb_=iw_rvb_[0]))
    phi_loss = -torch.sum(controlled_score.detach() * log_q_z_grad, dim=-1)
    return phi_loss


def _elbo(*, log_p_x__z: torch.Tensor, log_zeta: torch.Tensor, log_s: torch.Tensor):
    """Compute the ELBO."""
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s + log_zeta, dim=-1, keepdim=True)
    log_v_ = torch.where(torch.isinf(log_v), torch.zeros_like(log_v), log_v)
    elbo = torch.sum(log_s.exp() * log_v_, dim=-1)
    return elbo


def _estimate_iw_rvb(*, log_p_x__z: torch.Tensor, log_zeta: torch.Tensor, log_s: torch.Tensor, alpha: float = 0.0):
    """Compute the self-normalized Rényi importance weighted bound."""
    log_s_norm = log_s.log_softmax(dim=-1)
    if alpha == 1.0:
        return _elbo(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s_norm)
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s_norm + log_zeta, dim=-1, keepdim=True)
    # avoid numerical issues: in the case $\alpha > 1$, we need to reverse the sign of the masked `log_v` (`inf`).
    log_v = torch.where(log_v.isinf() & (log_v > 0), -log_v, log_v)
    self_normalized_bound = 1 / (1 - alpha) * torch.logsumexp(log_s_norm + (1 - alpha) * log_v, dim=-1)
    return self_normalized_bound
