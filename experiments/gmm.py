from __future__ import annotations

import argparse
import math
import typing
from typing import Any, Optional

import matplotlib.pyplot as plt
import pydantic
import rich
import rich.progress
import seaborn as sns
import torch
import torch.nn
from rich.progress import track

import vod
import wandb
from vod.gradients import estimate_iwrvb

from .utils.gmm import GaussianMixtureModel

sns.set()


class Args(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    train_steps: int = 100_000
    batch_size: int = 100
    n_samples: int = 20
    alpha: float = 0.0
    lr: float = 1e-3
    exp_version: str = "v0.1"
    device: str = "cpu"
    sampler: str = "priority"
    objective: str = "inbatch"
    grads_n_samples: int = 20
    wandb_project: str = "vod-gmm"

    @classmethod
    def parse(cls) -> "Args":
        """Parse arguments using `argparse`."""
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))


def evaluate_model(
    data: torch.Tensor,
    model: GaussianMixtureModel,
    objective: typing.Literal["vod-rws", "vod-ovis", "ovis", "inbatch"] = "vod-rws",
    sampler: typing.Literal["topk", "multinomial", "priority"] = "priority",
    alpha: float = 0.0,
    n_samples: int = 8,
) -> tuple[vod.VariationalObjective, dict[str,Any]]:
    """Evaluate the model on the given data."""
    objective_fn = {
        "vod-rws": vod.vod_rws,
        "vod-ovis": vod.vod_ovis,
        "ovis": vod.ovis,
        "inbatch": inbatch_lkl,
    }[objective]

    # forward pass
    out = model(data, n_samples=n_samples, sampler=sampler)

    # compute the probabilities
    z = out["z"]
    log_p_x__z = out["px"].log_prob(data[:, None, :]).sum(-1)
    log_p = out["pz"].logits.log_softmax(dim=-1)[:, None, :]
    log_p_z = log_p.expand(*z.shape, -1).gather(-1, z[..., None]).squeeze(-1)
    log_q____x = out["qz"].logits.log_softmax(dim=-1)[:, None, :]
    log_q_z__x = log_q____x.expand(*z.shape, -1).gather(-1, z[..., None]).squeeze(-1)
    log_s = out["log_s"]

    return objective_fn(
        log_p_x__z=log_p_x__z,
        f_theta=log_p_z,
        f_phi=log_q_z__x,
        log_s=log_s,
        alpha=alpha,
    ), {k:v for k, v in out.items() 
        if k in ["prior_kl", "posterior_kl", "logp_true", "logp_model"]}


def run(args: Args) -> dict[str, Any]:
    """Run the experiment."""
    args = Args.parse()
    rich.print(args)
    wandb.init(project=args.wandb_project)
    wandb.config.update(args)

    model = GaussianMixtureModel()
    if torch.cuda.is_available():
        model = model.cuda()

    static_batch = model.sample_from_prior(n=3, from_optimal=True)["px"].sample()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    alpha = args.alpha
    for i in track(range(args.train_steps), description="Training..."):
        optimizer.zero_grad()
        px_true = model.sample_from_prior(n=args.batch_size, from_optimal=True)["px"]
        batch = px_true.sample()  # <- TODO
        objective, meta = evaluate_model(
            batch,
            model,
            n_samples=args.n_samples,
            objective=args.objective,
            sampler=args.sampler,
            alpha=alpha,
        )
        objective.loss.mean().backward()
        optimizer.step()
        wandb.log(
            {
                "train/alpha": alpha,
                "train/loss": objective.loss.mean().cpu().detach(),
                **{f"train/{k}": v.mean().cpu().detach() for k, v in objective.diagnostics().items()},
                **{f"train/{k}": v.mean().cpu().detach() for k, v in meta.items()},
            },
            step=i,
        )
        if i % 5_000 == 0:
            _log_prior(model, i)
            _log_posterior(model, static_batch, i)


def inbatch_lkl(  # noqa: PLR0913
    log_p_x__z: torch.Tensor,
    f_theta: torch.Tensor,
    f_phi: torch.Tensor,
    f_psi: Optional[torch.Tensor] = None,
    log_s: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
) -> vod.VariationalObjective:
    """VOD-RWS method."""
    if f_psi is None:
        f_psi = f_phi
    log_zeta = f_theta - f_psi
    if log_s is None:
        log_s = torch.zeros_like(f_psi) - math.log(f_psi.shape[-1])

    # compute the IWB
    with torch.no_grad():
        logp, elbo, iw_rvb = [
            estimate_iwrvb(
                log_p_x__z=log_p_x__z,
                log_zeta=log_zeta,
                log_s=log_s,
                alpha=a,
            )
            for a in [0, 1, alpha]
        ]

    log_pz = f_theta.log_softmax(dim=-1)
    log_qz = f_phi.log_softmax(dim=-1)

    # compute the loss for `theta`
    # theta_loss = -torch.logsumexp(log_p_x__z + log_pz, dim=-1)
    log_s_ = log_s.log_softmax(dim=-1).detach()
    log_w = log_p_x__z + log_pz - log_qz.detach()
    theta_loss = - torch.logsumexp(log_s_ + (1-alpha) * log_w , dim=-1)

    # compute the loss for `phi` : KL(p(z|x) || q(z | x))
    log_p_z__x = (log_p_x__z + log_pz - logp[..., None]).detach()
    phi_loss = -torch.sum(log_p_z__x.softmax(dim=-1) * log_qz, dim=-1)

    return vod.VariationalObjective(logp=logp, elbo=elbo, iwrvb=iw_rvb, theta_loss=theta_loss, phi_loss=phi_loss)


@torch.no_grad()
def _log_prior(model: GaussianMixtureModel, step: int) -> None:
    """Plot the learned and optimal latent distributions."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        model.p_mu.cpu().detach(),
        model.log_theta.softmax(dim=-1).view(-1).cpu().detach().numpy(),
        label="learned",
    )
    plt.plot(
        model.p_mu.cpu().detach(),
        model.log_theta_opt.softmax(dim=-1).view(-1).cpu().detach().numpy(),
        label="optimal",
    )
    plt.legend()
    # log the plot to wandb
    wandb.log({"prior": wandb.Image(plt)}, step=step)
    plt.close()


@torch.no_grad()
def _log_posterior(model: GaussianMixtureModel, static_batch: torch.Tensor, step: int) -> None:
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("husl", n_colors=len(static_batch))
    for j, x in enumerate(static_batch):
        approx_p = model.approx_posterior(x[None]).logits[0]
        true_p = model.true_posterior(x[None]).logits[0]
        plt.axvline(x=x[0].cpu().detach(), color=colors[j], linestyle=":")
        plt.plot(
            model.p_mu.cpu(),
            approx_p.softmax(dim=-1).cpu().detach().numpy(),
            label="approx",
            color=colors[j],
        )
        plt.plot(
            model.p_mu.cpu(),
            true_p.softmax(dim=-1).cpu().detach().numpy(),
            label="true",
            color=colors[j],
            linestyle="--",
        )
    plt.legend()
    wandb.log({"posterior": wandb.Image(plt)}, step=step)
    plt.close()


if __name__ == "__main__":
    args = Args.parse()
    run(args)
