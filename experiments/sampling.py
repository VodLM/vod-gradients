import argparse

import numpy as np
import rich
import torch
import pydantic
import matplotlib.pyplot as plt
import seaborn as sns

import vod

sns.set()


class Args(pydantic.BaseModel):
    n_trials: int = 5_000
    n_samples_min: int = 1
    n_samples_max: int = 95
    n_samples_step: int = 10
    world_size: int = 100
    noise_scale_min: float = 1e-1
    noise_scale_max: float = 3.0
    noise_scale_step: int = 3
    tensor_type: str = "torch"
    a: float = -3
    b: float = 3

    class Config:
        extra = pydantic.Extra.forbid

    @classmethod
    def parse(cls) -> "Args":
        """Parse arguments using `argparse`"""
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))


def generate_axis(
    world_size: int,
    a: float = -3,
    b: float = 3,
) -> torch.Tensor:
    return torch.linspace(a, b, world_size)


def generate_scores(
    *,
    loc: float = 0,
    scale: float = 1.0,
    axis: torch.Tensor,
) -> torch.Tensor:
    log_p = torch.distributions.Normal(loc, scale).log_prob(axis)
    return log_p


@torch.no_grad()
def run():
    args = Args.parse()
    rich.print(args)

    axis = generate_axis(args.world_size, args.a, args.b)
    samplers = {
        "monte-carlo": vod.mc_sample,
        "priority": vod.priority_sample,
        "sn-priority": vod.priority_sample,
    }
    fig, axes = plt.subplots(
        figsize=(3 * args.noise_scale_step, 3 * 4),
        ncols=args.noise_scale_step,
        nrows=4,
        sharey="row",
    )
    colors = sns.color_palette()

    sample_range = np.logspace(
        np.log10(args.n_samples_min),
        np.log10(args.n_samples_max),
        num=args.n_samples_step,
        base=10,
        dtype=np.int32,
    )
    sample_range = np.unique(sample_range)
    scale_range = np.logspace(
        np.log10(args.noise_scale_min),
        np.log10(args.noise_scale_max),
        num=args.noise_scale_step,
        dtype=np.float64,
    )
    scale_range = np.unique(scale_range)
    for i, noise_scale in enumerate(scale_range):
        f_values = generate_scores(loc=0, scale=noise_scale, axis=axis)
        probs = f_values.softmax(dim=-1)

        h_values = probs + torch.randn_like(f_values)
        expected_value = (h_values * probs).sum(dim=-1)

        for j, (sampler_name, sampler) in enumerate(samplers.items()):
            data = {"mean": [], "upper": [], "lower": [], "var": [], "bias": []}
            for k, n_samples in enumerate(sample_range):
                f_values_ = f_values[None, :].expand(args.n_trials, -1)
                if args.tensor_type == "torch":
                    samples = sampler(f_values_, n_samples=int(n_samples))
                elif args.tensor_type == "numpy":
                    samples = sampler(f_values_.numpy(), n_samples=int(n_samples))
                    samples = samples.to(torch.Tensor)
                else:
                    raise ValueError(f"Unknown tensor type: {args.tensor_type}")

                if sampler_name.startswith("sn-"):
                    w = samples.log_weights.softmax(dim=-1)
                else:
                    w = samples.log_weights.exp()

                mc_estimates = (w * h_values[samples.indices]).sum(dim=-1)
                expected_h = (mc_estimates - expected_value).abs()
                data["mean"].append(mc_estimates.mean().item())
                data["upper"].append(mc_estimates.quantile(0.95).item())
                data["lower"].append(mc_estimates.quantile(0.05).item())
                data["var"].append((mc_estimates - expected_h).var().item())
                data["bias"].append(expected_h.mean().item())

            # data
            if j == 0:
                axes[0, i].plot(axis, probs, label="$p(z)$", color="black")
                axes[0, i].plot(axis, h_values.softmax(dim=-1), label=r"$\mathrm{softmax}(h(z))$", color="gray")
                axes[0, i].set_title(rf"$p(z) = N(0, {noise_scale:.1f}^2)$")
                axes[0, 0].set_ylabel(r"data")
                if i == len(scale_range) - 1:
                    axes[0, i].legend()

            # estimates
            axes[1, i].set_xscale("log")
            axes[1, 0].set_ylabel(r"estimates")
            if j == 0:
                axes[1, i].fill_between(sample_range, data["lower"], data["upper"], alpha=0.1, color=colors[j])
            axes[1, i].plot(sample_range, data["lower"], color=colors[j], alpha=0.5, linestyle="--")
            axes[1, i].plot(sample_range, data["upper"], color=colors[j], alpha=0.5, linestyle="--")
            axes[1, i].plot(sample_range, data["mean"], label=sampler_name, color=colors[j])
            if i == len(scale_range) - 1:
                axes[1, i].legend()

            # variance
            axes[2, 0].set_xscale("log")
            axes[2, 0].set_yscale("log")
            axes[2, 0].set_ylabel("variance")
            if j == 0:
                axes[2, i].plot(
                    sample_range,
                    np.power(sample_range, -1.0),
                    label=r"$K^{-1}$",
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                )
            axes[2, i].plot(sample_range, data["var"], label=sampler_name, color=colors[j])
            if i == len(scale_range) - 1:
                axes[2, i].legend()

            # bias
            axes[3, 0].set_xscale("log")
            axes[3, 0].set_yscale("log")
            axes[3, 0].set_ylabel("bias")
            axes[3, i].set_xlabel("number of samples")
            if j == 0:
                axes[3, i].plot(
                    sample_range,
                    np.power(sample_range, -0.5),
                    label=r"$K^{-1/2}$",
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                )
            axes[3, i].plot(sample_range, data["bias"], label=sampler_name, color=colors[j])
            if i == len(scale_range) - 1:
                axes[3, i].legend()

    plt.tight_layout()
    plt.savefig(".assets/mc_convergence.png")
    plt.close()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    run()
