from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from typing import Generic, TypeVar, Union, Any, Callable, Protocol

from typing_extensions import Type

Ts = TypeVar("Ts", bound=Union[torch.Tensor, np.ndarray])
Ts_o = TypeVar("Ts_o", bound=Union[torch.Tensor, np.ndarray])


@dataclasses.dataclass
class Samples(Generic[Ts]):
    indices: Ts
    log_weights: Ts

    def __iter__(self) -> "Samples[Ts]":
        for i in range(self.indices.shape[0]):
            yield Samples(self.indices[i], self.log_weights[i])

    def __getitem__(self, item: Any) -> "Samples[Ts]":
        return Samples(self.indices[item], self.log_weights[item])

    def __len__(self) -> int:
        return self.indices.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.indices.shape)

    def __repr__(self):
        return (
            f"Samples[{type(self.indices).__name__}]"
            f"(indices={self.indices.shape}, log_weights={self.log_weights.shape})"
        )

    def to(self, target: Type[Ts_o]) -> "Samples[Ts_o]":
        if target == np.ndarray:
            if isinstance(self.indices, torch.Tensor):
                return Samples(
                    indices=self.indices.detach().cpu().numpy(),
                    log_weights=self.log_weights.detach().cpu().numpy(),
                )
            elif isinstance(self.indices, np.ndarray):
                return self
            else:
                raise TypeError(f"Unknown type for indices: {type(self.indices)}")
        elif target == torch.Tensor:
            if isinstance(self.indices, torch.Tensor):
                return self
            elif isinstance(self.indices, np.ndarray):
                return Samples(
                    indices=torch.from_numpy(self.indices),
                    log_weights=torch.from_numpy(self.log_weights),
                )
            else:
                raise TypeError(f"Unknown type for indices: {type(self.indices)}")
        else:
            raise TypeError(f"Unknown target type: {target}")


class SampleFn(Protocol[Ts]):
    def __call__(self, scores: Ts, n_samples: int) -> Samples[Ts]:
        ...


class RunAsTorch:
    def __init__(self, fn: Callable[[Ts, ...], Samples[Ts]]):
        self.fn = fn

    def __call__(self, scores: Ts, n_samples: int) -> Samples[Ts]:
        if isinstance(scores, np.ndarray):
            return self.fn(torch.from_numpy(scores), n_samples).to(np.ndarray)
        elif isinstance(scores, torch.Tensor):
            return self.fn(scores, n_samples)
        else:
            raise TypeError(f"Unknown type for scores: {type(scores)}")


def multinomial_sample(scores: Ts, n_samples: int) -> Samples[Ts]:
    scores_type = type(scores)
    sample_fn = {
        torch.Tensor: _mc_sample_torch,
        np.ndarray: RunAsTorch(_mc_sample_torch),
    }[scores_type]
    return sample_fn(scores, n_samples)


def priority_sample(scores: Ts, n_samples: int, normalize: bool = True) -> Samples[Ts]:
    scores_type = type(scores)
    sample_fn = {
        torch.Tensor: _priority_sample_torch,
        np.ndarray: RunAsTorch(_priority_sample_torch),
    }[scores_type]

    return sample_fn(scores, n_samples, normalize=normalize)


def topk_sample(scores: Ts, n_samples: int) -> Samples[Ts]:
    scores_type = type(scores)
    sample_fn = {
        torch.Tensor: _topk_sample_torch,
        np.ndarray: RunAsTorch(_topk_sample_torch),
    }[scores_type]

    return sample_fn(scores, n_samples)


def _mc_sample_torch(scores: torch.Tensor, n_samples: int) -> Samples[torch.Tensor]:
    probs = torch.softmax(scores, dim=-1)
    indices = torch.multinomial(probs, n_samples, replacement=True)
    weights = torch.zeros_like(indices, dtype=torch.float32) - math.log(indices.shape[-1])
    return Samples(indices, weights)


def _priority_sample_torch(
    scores: torch.Tensor,
    n_samples: int,
    mode: str = "uniform",
    normalize: bool = True,
) -> Samples[torch.Tensor]:
    """Sample a distributions using priority sampling."""
    log_p = scores.log_softmax(dim=-1)

    if mode == "uniform":
        u = torch.rand_like(log_p).clamp(min=1e-12)
    elif mode == "exponential":
        u = torch.empty_like(log_p)
        u.exponential_()
    else:
        raise ValueError(f"Unknown mode `{mode}`")

    log_u = u.log()
    keys = log_p - log_u
    indices = keys.argsort(dim=-1, descending=True)[..., : n_samples + 1]
    if n_samples < scores.shape[-1]:
        tau = indices[..., -1:]
        log_tau = keys.gather(dim=-1, index=tau)[..., :1]
    else:
        log_tau = -float("inf") + torch.zeros_like(log_p)
    indices = indices[..., :n_samples]
    log_pi = log_p.gather(dim=-1, index=indices)

    if mode == "uniform":
        log_qz = torch.where(log_pi - log_tau < 0, log_pi - log_tau, torch.zeros_like(log_pi))
    elif mode == "exponential":
        log_qz = log_pi - log_tau
        log_qz = (log_qz).exp().mul(-1).exp().mul(-1).log1p()
    else:
        raise ValueError(f"Unknown mode `{mode}`")

    # finally, compute the log weights
    log_weights = log_pi - log_qz

    if normalize:
        log_weights = log_weights.log_softmax(dim=-1)

    return Samples(indices=indices, log_weights=log_weights)


def _topk_sample_torch(scores: torch.Tensor, n_samples: int) -> Samples[torch.Tensor]:
    """Sample the top-K values."""
    indices = scores.argsort(dim=-1, descending=True)[..., :n_samples]
    log_weight = scores.gather(dim=-1, index=indices)
    log_weight = log_weight.log_softmax(dim=-1)

    return Samples(indices=indices, log_weights=log_weight)
