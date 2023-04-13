import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class VOD:
    log_p: torch.Tensor
    theta_loss: torch.Tensor
    phi_loss: Optional[torch.Tensor] = None


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

    if log_s is None:
        log_s = torch.zeros_like(log_q_z)
    log_zeta = log_p_z - log_q_z
    log_v_ = log_p_x__z + log_zeta - torch.logsumexp(log_s + log_zeta, dim=-1, keepdim=True)

    # compute the approximate IW-RVB (VOD)
    self_normalized_bound = 1 / (1 - alpha) * torch.logsumexp(log_s + (1 - alpha) * log_v_, dim=-1)

    # compute the gradients
    log_p_z_grad_ = log_p_z - torch.sum(
        torch.softmax(log_s + log_zeta, dim=-1).detach() * log_zeta,
        dim=-1,
        keepdim=True,
    )

    w = torch.softmax(log_s + (1 - alpha) * (log_p_x__z + log_zeta), dim=-1)
    loss = -torch.sum(w.detach() * (log_p_x__z + log_p_z_grad_), dim=-1)

    return VOD(log_p=self_normalized_bound, theta_loss=loss, phi_loss=None)
