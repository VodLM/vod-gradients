import argparse
import collections
import math
from typing import Callable, Optional, Iterable

import numpy as np
import pydantic
import rich.progress
import torch.nn
import torchvision

from torchvision import datasets, transforms
import vod

import loguru
import wandb


def linear_warmup(step: float, period: float, start: float, end: float) -> float:
    """Linearly warm up from `start` to `end` over `period` steps"""
    t = max(0.0, min(step / period, 1.0))
    return start + t * (end - start)


class Args(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    num_epochs: int = 20
    batch_size: int = 64
    n_samples: int = 32
    latent_dim: int = 8
    latent_size: int = 128
    embedding_dim: int = 32
    alpha_start: float = 0.0
    alpha: float = 0.0
    warmup_steps: float = 5_000
    lr: float = 1e-3
    exp_version: str = "v0.1"
    device: str = "cpu"
    sampler: str = "priority"

    @classmethod
    def parse(cls) -> "Args":
        """Parse arguments using `argparse`"""
        parser = argparse.ArgumentParser()
        for field in cls.__fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

        args = parser.parse_args()
        return cls(**vars(args))


class ToBinary:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.bool()


class Dataset:
    def __init__(self, root="data", **kwargs):
        self.train = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=self._transform,
        )

        self.test = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=self._transform,
        )

    @property
    def _transform(self) -> Callable:
        return transforms.Compose([transforms.ToTensor(), ToBinary()])

    def __repr__(self):
        return f"Dataset(train={len(self.train)}, test={len(self.test)})"


class ConditionalDiscreteVae(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 1_024,
        num_layers: int = 3,
        latent_dim: int = 8,
        latent_size: int = 32,
        embedding_dim: int = 32,
        output_shape: tuple[int, ...] = (1, 28, 28),
        num_labels: int = 10,
    ):
        super().__init__()

        # memory
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Parameter(torch.randn(latent_dim, latent_size, embedding_dim), requires_grad=True)

        # prior embeddings
        self.num_labels = num_labels
        self.label_embeddings = torch.nn.Embedding(num_labels, self.latent_dim * self.embedding_dim)

        # encoder
        layers = [
            torch.nn.Linear(int(math.prod(output_shape)), hidden_size),
            torch.nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                ]
            )
        layers.append(torch.nn.Linear(hidden_size, self.latent_dim * self.embedding_dim))
        self.encoder = torch.nn.Sequential(*layers)

        # decoder
        layers = [
            torch.nn.Linear(self.latent_dim * self.embedding_dim, hidden_size),
            torch.nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                ]
            )
        layers.append(torch.nn.Linear(hidden_size, int(math.prod(output_shape))))
        self.decoder = torch.nn.Sequential(*layers)

    @property
    def theta(self) -> Iterable[torch.Tensor]:
        return (p for k, p in self.named_parameters() if "decoder" in k)

    @property
    def phi(self) -> Iterable[torch.Tensor]:
        return (p for k, p in self.named_parameters() if "encoder" in k)

    def prior(self, y: torch.Tensor) -> torch.Tensor:
        h_y = self.label_embeddings(y)
        h_y = h_y.view(*h_y.shape[:-1], self.latent_dim, self.embeddings.shape[-1])
        logits = torch.einsum("...nd,nmd-> ...nm", h_y, self.embeddings)
        return logits.log_softmax(dim=-1)

    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        x_flatten = x.view(x.shape[0], -1).float()
        h_x = self.encoder(x_flatten)
        h_x = h_x.view(*h_x.shape[:-1], self.latent_dim, self.embedding_dim)
        logits = torch.einsum("...nd,nmd-> ...nm", h_x, self.embeddings)
        return logits.log_softmax(dim=-1)

    def sample(
        self,
        y: Optional[torch.Tensor] = None,
        n_samples: int = 10,
        sample_fn: vod.SampleFn = vod.priority_sample,
    ) -> torch.Tensor:
        if y is None:
            y = torch.range(0, self.num_labels - 1, device=self.embeddings.device).long()
        prior = self.prior(y)
        z = sample_fn(prior, n_samples=n_samples)
        return self._decode(z.indices)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 100,
        alpha: float = 0.0,
        sample_fn: vod.SampleFn = vod.priority_sample,
    ) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        prior = self.prior(y)
        posterior = self.posterior(x)
        z = sample_fn(posterior, n_samples=n_samples)
        log_p_z = prior.gather(dim=-1, index=z.indices)
        log_q_z = posterior.gather(dim=-1, index=z.indices)
        x_logits = self._decode(z.indices)
        p_x__z = torch.distributions.Bernoulli(logits=x_logits)
        log_p_x__z = p_x__z.log_prob(x.view(batch_size, -1)[:, None].float()).sum(dim=-1)
        vod_data = vod.vod_objective(
            log_p_x__z=log_p_x__z,
            log_s=z.log_weights.sum(dim=1),
            log_p_z=log_p_z.sum(dim=1),
            log_q_z=log_q_z.sum(dim=1),
            alpha=alpha,
        )

        return {
            "x_logits": x_logits,
            "log_p_z": log_p_z,
            "log_p_x__z": log_p_x__z,
            "loss": vod_data.loss,
            "logp": vod_data.log_p,
            "kl": vod_data.kl,
        }

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        embs = self.embeddings[None].expand(z.shape[0], *self.embeddings.shape)
        z_ = z[..., None].expand(*z.shape, embs.shape[-1])
        z_mem_items = embs.gather(dim=-2, index=z_)
        z_mem_items = z_mem_items.permute(0, 2, 1, 3).contiguous()
        z_embs = z_mem_items.view(*z_mem_items.shape[:2], -1)
        x_logits = self.decoder(z_embs)
        return x_logits


def run():
    args = Args.parse()
    rich.print(args)
    wandb.config.update(args)

    dataset = Dataset()
    rich.print(dataset)

    sample_fn = {
        "priority": vod.priority_sample,
        "multinomial": vod.multinomial_sample,
        "topk": vod.topk_sample,
    }[args.sampler]
    model = ConditionalDiscreteVae(
        latent_dim=args.latent_dim,
        latent_size=args.latent_size,
        embedding_dim=args.embedding_dim,
        num_labels=10,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dl = torch.utils.data.DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset.test, batch_size=args.batch_size, shuffle=True)
    train_metrics = collections.defaultdict(list)
    test_metrics = collections.defaultdict(list)
    global_step = 0
    track_keys = ["loss", "logp", "kl"]
    for e in range(args.num_epochs):
        for i, (x, y) in enumerate(rich.progress.track(train_dl, description="training")):
            alpha = linear_warmup(
                global_step,
                period=args.warmup_steps,
                start=args.alpha_start,
                end=args.alpha,
            )
            x = x.to(args.device)
            y = y.to(args.device)
            model.zero_grad()
            output = model(x, y, alpha=alpha, n_samples=args.n_samples, sample_fn=sample_fn)
            loss = output["loss"].mean()
            loss.backward()
            optimizer.step()
            global_step += 1
            for key in track_keys:
                train_metrics[key] += [output[key].detach().mean().item()]
            train_metrics["step"] += [global_step]
            if i % 100 == 0:
                grads = _compute_grads(model=model, x=x, y=y)
                grad_stats = {}
                for k, v in grads.items():
                    k_stats = _grads_stats(v)
                    for k2, v2 in k_stats.items():
                        grad_stats[f"grad/{k}/{k2}"] = v2
                wandb.log(grad_stats, step=global_step)

                with torch.inference_mode():
                    # plot the samples for the first element in the batch
                    _plot_prior_samples(model, global_step)
                    for key, value in _test_model(model, test_dl, track_keys=track_keys).items():
                        test_metrics[key] += [value]
                    test_metrics["step"] += [global_step]
                loguru.logger.info(
                    f"epoch={e} step={i} train.logp={train_metrics['logp'][-1]:.2f} test.logp={test_metrics['logp'][-1]:.2f}"
                )
                wandb.log(
                    {
                        "parameters/alpha": alpha,
                        **{f"train/{k}": v[-1] for k, v in train_metrics.items()},
                        **{f"test/{k}": v[-1] for k, v in test_metrics.items()},
                    },
                    step=global_step,
                )


def _log_prob(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(-1)
    rich.print(dict(log_probs=log_probs.shape, indices=indices.shape))
    return torch.gather(log_probs, -1, indices).squeeze(-1)


def _test_model(
    model: ConditionalDiscreteVae,
    test_dl: torch.utils.data.DataLoader,
    track_keys: list[str],
) -> Iterable[float]:
    metrics = collections.defaultdict(list)
    for j, (x, y) in enumerate(test_dl):
        x = x.to(model.embeddings.device)
        y = y.to(model.embeddings.device)
        if j > 10:
            break
        output = model(x, y)
        for key in track_keys:
            metrics[key] += [output[key].detach().mean().item()]
    return {k: np.mean(v) for k, v in metrics.items()}


def _compute_grads(
    *,
    model: ConditionalDiscreteVae,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.0,
    n_samples: int = 8,
) -> dict[str, torch.Tensor]:
    x = x.to(model.embeddings.device)
    y = y.to(model.embeddings.device)
    grads = {}
    for xi, yi in zip(x, y):
        model.zero_grad()
        output = model(xi[None], yi[None], alpha=alpha, n_samples=n_samples)
        loss = output["loss"].mean()
        loss.backward()
        for name, params in {"theta": model.theta, "phi": model.phi}.items():
            ngrads = (p.grad for p in params if p.grad is not None)
            ngrads = [g.view(1, -1) for g in ngrads]
            ngrads = torch.cat(ngrads, dim=1)
            if name not in grads:
                grads[name] = ngrads
            else:
                grads[name] = torch.cat([grads[name], ngrads])

    return grads


def _grads_stats(grads: torch.Tensor) -> dict[str, float]:
    grads = grads[grads.any(dim=1)]
    if grads.numel() == 0:
        return {}
    return {
        "mean": grads.mean(dim=1).abs().mean().item(),
        "std": grads.std(dim=1).abs().mean().item(),
    }


def _plot_prior_samples(model, global_step):
    logits = model.sample(n_samples=10).detach()
    logits = logits.reshape(10, 10, 28, 28).contiguous().cpu().numpy()
    grid = torchvision.utils.make_grid(
        torch.from_numpy(logits).view(-1, 1, 28, 28),
        nrow=10,
        normalize=True,
        range=(0, 1),
    )
    # save the grid as a png
    pil_image = torchvision.transforms.ToPILImage()(grid)
    pil_image.save(f"outputs/mnist_{global_step}.png")
    pil_image.save("outputs/mnist_latest.png")
    wandb.log({"samples/prior": wandb.Image(pil_image)}, step=global_step)


if __name__ == "__main__":
    wandb.init(project="grads")
    run()
    wandb.finish()
