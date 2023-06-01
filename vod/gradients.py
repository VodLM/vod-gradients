from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Optional, Protocol

import torch
import rich


@dataclasses.dataclass
class VariationalObjective:
    logp: torch.Tensor
    elbo: torch.Tensor
    iwrvb: torch.Tensor
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
            "iwrvb": self.iwrvb,
            "theta_loss": self.theta_loss,
            "phi_loss": self.phi_loss,
            "kl_vi": self.kl_vi,
        }


def _compute_iw_rvb(log_w_z: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    if alpha == 1:
        return torch.mean(log_w_z, dim=-1)

    return 1 / (1 - alpha) * (torch.logsumexp((1 - alpha) * log_w_z, dim=-1) - math.log(log_w_z.shape[-1]))


class VariationalObjectiveFn(Protocol):
    def __call__(
        log_p_x__z: torch.Tensor,
        f_theta: torch.Tensor,
        f_phi: torch.Tensor,
        f_psi: Optional[torch.Tensor] = None,
        log_s: Optional[torch.Tensor] = None,
        alpha: float = 0.0,
    ) -> VariationalObjective:
        ...


def ovis(
    f_phi: torch.Tensor,
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_psi: Optional[torch.Tensor] = None,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VariationalObjective:
    """Compute the VOD objective and its gradients."""
    log_p_z = f_theta.log_softmax(dim=-1)
    log_q_z = f_phi.log_softmax(dim=-1)
    log_w_z = log_p_x__z + log_p_z - log_q_z

    if log_s is not None and (log_s != log_s.view(-1)[0]).any():
        warnings.warn("OVIS objective assumes equal importance weights. The weights will be ignored.")

    with torch.no_grad():
        # compute the IWB
        logp, elbo, iw_rvb = [_compute_iw_rvb(log_w_z, alpha=a) for a in [0, 1, alpha]]

    # compute the loss for `theta`
    log_w_a = (1 - alpha) * log_w_z
    v_a = torch.softmax(log_w_a, dim=-1)
    theta_loss = -torch.sum(v_a.detach() * (log_p_x__z + log_p_z), dim=-1)

    # compute the loss for `phi`
    def _ovis_phi_loss(
        log_w_z: torch.Tensor, log_q_z: torch.Tensor, alpha: float = 0.0, eps: float = 1e-5
    ) -> torch.Tensor:
        if alpha == 0:
            return -torch.mean(log_w_z.detach() * log_q_z, dim=-1)

        with torch.no_grad():
            log_w_alp = (1 - alpha) * log_w_z
            v_a = torch.softmax(log_w_alp, dim=-1)
            k = math.log(log_w_z.shape[-1])
            v_a_ = v_a.clamp(max=1 - eps)
            controlled_score = -(torch.log1p(-v_a_) - math.log(1 - 1 / k))  # minus?

        return -torch.sum(controlled_score.detach() * log_q_z, dim=-1)

    phi_loss = _ovis_phi_loss(log_w_z, log_q_z, alpha=alpha)

    output = VariationalObjective(logp=logp, elbo=elbo, iwrvb=iw_rvb, theta_loss=theta_loss, phi_loss=phi_loss)

    return output


def vod_ovis(
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: Optional[torch.Tensor] = None,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VariationalObjective:
    r"""Compute the VOD objective and its gradients.

    Args:
        log_p_x__z: \propto log p(x|z), parameter `theta` (unnormalized)
        f_theta: \propto log p(z), parameter `theta` (unnormalized)
        f_phi: \propto log q(z|x) (approximate posterior, parameter `phi`) (unnormalized)
        f_psi: \propto log q(z|x) (sampling distribution, parameter `psi`) (unnormalized)
        log_s: importance weights for the samples `z ~ q_\psi(z|x)`
        alpha: Rényi alpha parameter

    Returns:
        VOD: VOD bound and its gradients
    """
    if f_psi is None:
        f_psi = f_phi
    log_zeta = f_theta - f_psi
    if log_s is None:
        log_s = torch.zeros_like(f_psi) - math.log(f_psi.shape[-1])

    # compute the IWB
    logp = estimate_iwrvb(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s,
        alpha=0,
    )

    # compute the ELBO
    elbo = estimate_iwrvb(
        log_p_x__z=log_p_x__z,
        log_zeta=log_zeta,
        log_s=log_s,
        alpha=1,
    )

    # compute the approximate IW-RVB (VOD)
    iw_rvb = estimate_iwrvb(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s, alpha=alpha)

    # compute the gradients for `\nabla_theta log p(z)`
    log_p_z_grad = f_theta - torch.sum(
        torch.softmax(log_s + log_zeta, dim=-1).detach() * f_theta,
        dim=-1,
        keepdim=True,
    )

    # compute the loss for `theta`
    w = torch.softmax(log_s + (1 - alpha) * (log_p_x__z + log_zeta), dim=-1)
    theta_loss = -torch.sum(w.detach() * (log_p_x__z + log_p_z_grad), dim=-1)

    # compute the loss for `phi`
    phi_loss = _vod_phi_loss(log_p_x__z, f_theta, f_phi, f_psi, log_s, alpha=alpha)

    return VariationalObjective(logp=logp, elbo=elbo, iwrvb=iw_rvb, theta_loss=theta_loss, phi_loss=phi_loss)


def _vod_phi_loss(
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: torch.Tensor,
    log_s_psi: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
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

        iw_rvb = estimate_iwrvb(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=los_s_phi, alpha=alpha)
        iw_rvb_ = estimate_iwrvb(log_p_x__z=log_p_x__z_, log_zeta=log_zeta_, log_s=los_s_phi_, alpha=alpha)
        controlled_score = iw_rvb[:, None] - iw_rvb_

    # compute the gradients for `\nabla_theta log p(z)`
    log_q_z_grad = f_phi - torch.sum(s_phi.detach() * f_phi, dim=-1, keepdim=True)
    # rich.print(dict(log_q_z_grad=log_q_z_grad[0], controlled_score=controlled_score[0], log_p_x__z=log_p_x__z[0], log_p_x__z_=log_p_x__z_[0], iw_rvb=iw_rvb[0], iw_rvb_=iw_rvb_[0]))
    phi_loss = -torch.sum(controlled_score.detach() * log_q_z_grad, dim=-1)
    return phi_loss


def vod_rws(  # noqa: PLR0913
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: Optional[torch.Tensor] = None,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> VariationalObjective:
    """VOD-RWS method."""
    if f_psi is None:
        f_psi = f_phi
    log_zeta = f_theta - f_psi
    if log_s is None:
        log_s = torch.zeros_like(f_psi) - math.log(f_psi.shape[-1])

    # compute the IWB
    logp, elbo, iw_rvb = [
        estimate_iwrvb(
            log_p_x__z=log_p_x__z,
            log_zeta=log_zeta,
            log_s=log_s,
            alpha=a,
        )
        for a in [0, 1, alpha]
    ]

    # compute the gradients for `\nabla_theta log p(z)`
    log_p_z_grad = f_theta - torch.sum(
        torch.softmax(log_s - f_psi + f_theta, dim=-1).detach() * f_theta, dim=-1, keepdim=True
    )

    # compute the loss for `theta`
    v_alpha = torch.softmax(log_s + (1 - alpha) * (log_p_x__z + log_zeta), dim=-1)
    theta_loss = -torch.sum(v_alpha.detach() * (log_p_x__z + log_p_z_grad), dim=-1)

    # compute the gradients for `\nabla_theta log p(z)`
    log_q_z_grad = f_phi - torch.sum(
        torch.softmax(log_s - f_psi + f_phi, dim=-1).detach() * f_phi, dim=-1, keepdim=True
    )

    # compute the loss for `phi`
    v_zero = torch.softmax(log_s + log_p_x__z + log_zeta, dim=-1)
    phi_loss = -torch.sum(v_zero.detach() * log_q_z_grad, dim=-1)

    return VariationalObjective(logp=logp, elbo=elbo, iwrvb=iw_rvb, theta_loss=theta_loss, phi_loss=phi_loss)


@torch.jit.script
def _estimate_elbo(
    *,
    log_p_x__z: torch.Tensor,
    log_zeta: torch.Tensor,
    log_s: torch.Tensor,
) -> torch.Tensor:
    """Compute the ELBO."""
    log_v = log_p_x__z + log_zeta - torch.logsumexp(log_s + log_zeta, dim=-1, keepdim=True)
    log_v_ = torch.where(torch.isinf(log_v), torch.zeros_like(log_v), log_v)
    elbo = torch.sum(log_s.exp() * log_v_, dim=-1)
    return elbo

@torch.jit.script
def _estimate_iwrvb(
    *,
    log_p_x__z: torch.Tensor,
    log_zeta: torch.Tensor,
    log_s: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    log_s_norm = log_s.log_softmax(dim=-1)
    log_v_hat = log_p_x__z + log_zeta - torch.logsumexp(log_s_norm + log_zeta, dim=-1, keepdim=True)
    # avoid numerical issues: in the case $\alpha > 1$, we need to reverse the sign of the masked `log_v` (`inf`).
    log_v_hat = torch.where(log_v_hat.isinf() & (log_v_hat > 0), -math.inf, log_v_hat)
    return (1 - alpha) ** (-1) * torch.logsumexp(log_s_norm + (1 - alpha) * log_v_hat, dim=-1)


def estimate_iwrvb(
    *,
    log_p_x__z: torch.Tensor,
    log_zeta: torch.Tensor,
    log_s: torch.Tensor,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Compute the self-normalized Rényi importance weighted bound."""
    if alpha == 1.0:  # noqa: PLR2004
        return _estimate_elbo(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s)
    
    return _estimate_iwrvb(log_p_x__z=log_p_x__z, log_zeta=log_zeta, log_s=log_s, alpha=alpha)
