import argparse
import collections
import math
from typing import Callable, Optional, Iterable

import numpy as np
import plotext as pltxt
import pydantic
import rich.progress
import torch.nn
import torchvision

from torchvision import datasets, transforms
import vod


class Args(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.forbid

    num_epochs: int = 20
    batch_size: int = 32
    n_samples: int = 32
    num_latents: int = 1_024
    embedding_dim: int = 32
    alpha: float = 0.0
    lr: float = 1e-4

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
    def __init__(self, root="data", n_memory: int = 1000, **kwargs):
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
        num_latents: int = 1_024,
        embedding_dim: int = 32,
        output_shape: tuple[int, ...] = (1, 28, 28),
        num_labels: int = 10,
    ):
        super().__init__()

        # memory
        self.memory = torch.nn.Parameter(torch.randn(num_latents, embedding_dim), requires_grad=True)

        # prior embeddings
        self.num_labels = num_labels
        self.label_embeddings = torch.nn.Embedding(num_labels, self.memory.shape[-1])

        # encoder
        layers = [torch.nn.Linear(int(math.prod(output_shape)), hidden_size), torch.nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                ]
            )
        layers.append(torch.nn.Linear(hidden_size, embedding_dim))
        self.encoder = torch.nn.Sequential(*layers)

        # decoder
        layers = [torch.nn.Linear(self.memory.shape[-1], hidden_size), torch.nn.GELU()]
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

    def prior(self, y: torch.Tensor) -> torch.distributions.Categorical:
        h_y = self.label_embeddings(y)
        logits = torch.einsum("id,md-> im", h_y, self.memory)
        return torch.distributions.Categorical(logits=logits)

    def posterior(self, x: torch.Tensor) -> torch.distributions.Categorical:
        x_flatten = x.view(x.shape[0], -1).float()
        h_x = self.encoder(x_flatten)
        x_logits = torch.einsum("id,md-> im", h_x, self.memory)
        return torch.distributions.Categorical(logits=x_logits)

    def sample(self, y: Optional[torch.Tensor] = None, n_samples: int = 10) -> torch.Tensor:
        if y is None:
            y = torch.range(0, self.num_labels - 1, device=self.memory.device).long()
        prior = self.prior(y)
        z = prior.sample(torch.Size((n_samples,)))
        return self._decode(z)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 100, alpha: float = 0.0
    ) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        prior = self.prior(y)
        posterior = self.posterior(x)
        z = posterior.sample(torch.Size((n_samples,)))
        log_p_z = prior.log_prob(z)
        log_q_z = posterior.log_prob(z)
        x_logits = self._decode(z)
        p_x__z = torch.distributions.Bernoulli(logits=x_logits)
        log_p_x__z = p_x__z.log_prob(x.view(batch_size, -1)[None, ...].float()).sum(dim=-1)
        vod_data = vod.vod_objective(
            log_p_x__z=log_p_x__z.T,
            log_p_z=log_p_z.T,
            log_q_z=log_q_z.T,
            alpha=alpha,
        )

        return {
            "x_logits": x_logits,
            "log_p_z": log_p_z,
            "log_p_x__z": log_p_x__z,
            "loss": vod_data.theta_loss,
            "bound": vod_data.log_p,
        }

    def _decode(self, z):
        z_mem_items = self.memory[z]
        x_logits = self.decoder(z_mem_items)
        return x_logits


def run():
    args = Args.parse()
    rich.print(args)

    dataset = Dataset()
    rich.print(dataset)

    model = ConditionalDiscreteVae(
        num_latents=args.num_latents,
        embedding_dim=args.embedding_dim,
        num_labels=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dl = torch.utils.data.DataLoader(dataset.train, batch_size=args.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset.test, batch_size=args.batch_size, shuffle=True)
    train_metrics = collections.defaultdict(list)
    test_metrics = collections.defaultdict(list)
    global_step = 0
    for e in range(args.num_epochs):
        for i, (x, y) in enumerate(rich.progress.track(train_dl, description="training")):
            output = model(x, y, alpha=args.alpha, n_samples=args.n_samples)
            loss = output["loss"].mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            train_metrics["bound"] += [output["bound"].mean().item()]
            train_metrics["step"] += [global_step]
            if i % 100 == 0:
                with torch.inference_mode():
                    # plot the samples for the first element in the batch
                    _plot_prior_samples(model, global_step)
                    test_likelihoods = list(_test_model(model, test_dl))
                    test_metrics["bound"] += [np.mean(test_likelihoods)]
                    test_metrics["step"] += [global_step]
                pltxt.clear_figure()
                pltxt.theme("dark")
                pltxt.plot(train_metrics["step"], train_metrics["bound"], label="train")
                pltxt.plot(test_metrics["step"], test_metrics["bound"], label="test")
                pltxt.show()


def _test_model(model: ConditionalDiscreteVae, test_dl: torch.utils.data.DataLoader) -> Iterable[float]:
    for j, (x, y) in enumerate(test_dl):
        if j > 10:
            break
        output = model(x, y)
        yield output["bound"].mean().item()


def _plot_prior_samples(model, global_step):
    logits = model.sample(n_samples=10).detach()
    logits = logits.permute(1, 0, 2).reshape(10, 10, 28, 28).contiguous().numpy()
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


if __name__ == "__main__":
    run()
