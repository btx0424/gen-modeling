"""
Shared rectified / linear flow-matching building blocks (parameterization wrappers,
MSE objectives, Euler sampling, EMA).

Used by image / trajectory examples and can pair with any denoiser
``forward(x_t, t, cond=None)``: optional ``cond`` is class labels (indices), a latent
vector (VFM), or omitted for fully unconditional backbones.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

PredictionType = Literal["x", "eps", "v"]
LossType = Literal["x", "eps", "v"]
ModelArch = Literal[
    "vanilla",
    "global_residual",
    "corrected_residual1",
    "corrected_residual2",
]


def compute_flow_matching_loss(
    loss_type: LossType,
    x1: torch.Tensor,
    eps: torch.Tensor,
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cond_steps: int = 0,
) -> torch.Tensor:
    x1_hat, v_hat, eps_hat = predictions
    if loss_type == "x":
        err = (x1_hat - x1) ** 2
    elif loss_type == "eps":
        err = (eps_hat - eps) ** 2
    elif loss_type == "v":
        v_target = x1 - eps
        err = (v_hat - v_target) ** 2
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return err[:, cond_steps:].mean()


class PredictionWrapper(nn.Module):
    """
    Bridge x / eps / v prediction targets for linear interpolation ``x_t = t x1 + (1-t) eps``.

    The backbone is always invoked as ``network(x_t, t, cond=cond)`` with ``cond=None``
    when unconditioned. Residual parameterizations are selected with ``arch``.
    """

    def __init__(
        self,
        network: nn.Module,
        pred_type: PredictionType,
        arch: ModelArch = "vanilla",
    ) -> None:
        super().__init__()
        self.network = network
        self.pred_type = pred_type
        self.arch: ModelArch = arch
        self.sample_shape = network.sample_shape

    def _call_network(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.network(x_t, t, cond=cond)

    def _apply_arch(self, raw: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        t_b = t.reshape(expand_shape)
        if self.arch == "vanilla":
            return raw
        if self.arch == "global_residual":
            return raw + x_t
        if self.arch == "corrected_residual1":
            return raw + x_t / (1.0 - t_b)
        if self.arch == "corrected_residual2":
            return (t_b * raw + x_t) / (1.0 - t_b)
        raise ValueError(f"Invalid model arch: {self.arch}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._call_network(x_t, t, cond)
        pred = self._apply_arch(raw, x_t, t)
        expand_shape = (-1,) + (x_t.ndim - 1) * (1,)
        t_b = t.reshape(expand_shape)
        if self.pred_type == "x":
            x1_hat = pred
            v_hat = (x1_hat - x_t) / (1.0 - t_b)
            eps_hat = (x_t - t_b * x1_hat) / (1.0 - t_b)
        elif self.pred_type == "eps":
            eps_hat = pred
            v_hat = (x_t - eps_hat) / t_b
            x1_hat = x_t + (1.0 - t_b) * v_hat
        else:
            v_hat = pred
            x1_hat = x_t + (1.0 - t_b) * v_hat
            eps_hat = x_t - t_b * v_hat
        return x1_hat, v_hat, eps_hat


def prediction_wrapper(
    network: nn.Module, pred_type: PredictionType, arch: ModelArch
) -> PredictionWrapper:
    return PredictionWrapper(network, pred_type, arch)


def prediction_wrapper_class(arch: ModelArch):
    """Backward-compatible: ``prediction_wrapper_class(arch)(net, pred_type)``."""

    def _factory(network: nn.Module, pred_type: PredictionType) -> PredictionWrapper:
        return PredictionWrapper(network, pred_type, arch)

    return _factory


class LinearFlow(nn.Module):
    """Rectified flow with optional classifier-free guidance (conditional path)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        noise_scale: float = 1.0,
        loss_type: LossType = "v",
        t_eps: float = 1e-2,
        conditional: bool = False,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.sample_shape = torch.Size(model.sample_shape)
        self.noise_scale = noise_scale
        self.loss_type = loss_type
        self.t_eps = t_eps
        self.conditional = conditional
        self.class_dropout_prob = class_dropout_prob

    def _null_class_idx(self) -> int:
        inner = getattr(self.model, "network", None)
        backbone = getattr(inner, "backbone", None) if inner is not None else None
        if backbone is None or not hasattr(backbone, "null_class_idx"):
            raise TypeError("CFG / label dropout requires backbone.null_class_idx on the denoiser")
        return int(backbone.null_class_idx)

    def maybe_drop_cond(self, cond: torch.Tensor) -> torch.Tensor:
        if self.class_dropout_prob <= 0:
            return cond
        drop_mask = torch.rand(cond.shape[0], device=cond.device) < self.class_dropout_prob
        if not drop_mask.any():
            return cond
        out = cond.clone()
        out[drop_mask] = self._null_class_idx()
        return out

    def compute_loss(
        self,
        x1: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        cond_steps: int | None = None,
    ) -> torch.Tensor:
        if self.conditional and cond is None:
            raise ValueError("conditional flow: cond (class labels) is required for compute_loss")
        if not self.conditional and cond is not None:
            raise ValueError("unconditional flow: did not expect cond")

        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t = t.clip(self.t_eps, 1.0 - self.t_eps)
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t_view = t.reshape(expand_shape)
        eps = torch.randn_like(x1) * self.noise_scale
        if cond_steps is not None:
            if not (1 <= cond_steps <= x1.shape[1]):
                raise ValueError(
                    f"cond_steps must be in [1, T], got cond_steps={cond_steps}, T={x1.shape[1]}"
                )
            eps[:, :cond_steps] = x1[:, :cond_steps]
        x_t = t_view * x1 + (1.0 - t_view) * eps
        if cond_steps is not None:
            x_t[:, :cond_steps] = x1[:, :cond_steps]

        if self.conditional:
            assert cond is not None
            cond_in = self.maybe_drop_cond(cond)
            predictions = self.model(x_t, t, cond=cond_in)
        else:
            predictions = self.model(x_t, t, cond=None)

        return compute_flow_matching_loss(
            self.loss_type,
            x1=x1,
            eps=eps,
            predictions=predictions,
            cond_steps=cond_steps,
        )

    @torch.inference_mode()
    def sample(self, num_samples: int, device: torch.device, num_steps: int) -> torch.Tensor:
        if self.conditional:
            raise TypeError("use sample_cfg(...) for class-conditional models")
        dtype = next(self.model.parameters()).dtype
        x_t = torch.randn((num_samples,) + self.sample_shape, device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((num_samples,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, cond=None)
            x_t = x_t + v_hat * dt
        return x_t

    @torch.inference_mode()
    def sample_cond_prefix(
        self,
        cond_prefix: torch.Tensor,
        device: torch.device,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Sample a full window ``(N, T, D)`` with the first ``k`` timesteps fixed to
        ``cond_prefix`` (normalized space), matching the ``cond_steps`` loss path.
        """
        if self.conditional:
            raise TypeError("sample_cond_prefix is for unconditional flows with prefix conditioning")
        dtype = next(self.model.parameters()).dtype
        k = cond_prefix.shape[1]
        n = cond_prefix.shape[0]
        if cond_prefix.ndim != 3:
            raise ValueError(f"cond_prefix must be (N, k, D), got shape {tuple(cond_prefix.shape)}")
        seq_len, dim = self.sample_shape
        if cond_prefix.shape[2] != dim:
            raise ValueError(
                f"cond_prefix dim {cond_prefix.shape[2]} != sample_shape[1] {dim}"
            )
        if k > seq_len:
            raise ValueError(f"cond_steps k={k} exceeds seq_len={seq_len}")
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        x_t = torch.randn((n,) + self.sample_shape, device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        x_t[:, :k] = cond_prefix
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, cond=None)
            v_hat = v_hat.clone()
            v_hat[:, :k] = 0.0
            x_t = x_t + v_hat * dt
            x_t[:, :k] = cond_prefix
        return x_t

    @torch.inference_mode()
    def sample_cfg(
        self,
        cond: torch.Tensor,
        device: torch.device,
        num_steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        if not self.conditional:
            raise TypeError("sample_cfg requires a conditional flow (conditional=True)")
        dtype = next(self.model.parameters()).dtype
        n = cond.shape[0]
        x_t = torch.randn((n,) + self.sample_shape, device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        null = self._null_class_idx()
        uncond = torch.full_like(cond, null)
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_cond, _ = self.model(x_t, t, cond=cond)
            _, v_uncond, _ = self.model(x_t, t, cond=uncond)
            v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
            x_t = x_t + v_hat * dt
        return x_t


class VariationalFlow(nn.Module):
    """
    Variational flow matching: a latent ``z`` modulates the rectified-flow vector field.

    Training samples a flow time ``t``, builds the linear bridge ``x_t = t x1 + (1-t) eps``
    (with optional pinned prefix frames via ``cond_steps``), draws ``z ~ q(z|x1)`` from the
    encoder's Gaussian head, and trains the wrapped denoiser (``PredictionWrapper`` +
    backbone) with ``cond=z`` to match the usual flow-matching target. A weighted Gaussian
    KL term on the encoder output is added to the flow loss.

    The encoder must define ``latent_dim`` and return ``(mu, logvar)`` from ``forward``.
    Sampling draws ``z ~ N(0, I)`` and integrates the learned velocity field with that
    fixed ``z`` (see ``sample`` and ``sample_cond_prefix``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        model: nn.Module,
        *,
        noise_scale: float = 1.0,
        loss_type: LossType = "v",
        t_eps: float = 1e-2,
    ):
        super().__init__()
        self.encoder = encoder
        
        try:
            self.latent_dim = self.encoder.latent_dim
        except AttributeError:
            raise ValueError("encoder must have a `latent_dim` attribute")
        
        self.model = model
        self.sample_shape = torch.Size(model.sample_shape)
        self.noise_scale = noise_scale
        self.loss_type = loss_type
        self.t_eps = t_eps
        self.conditional = True
    
    def compute_loss(
        self,
        x1: torch.Tensor,
        *,
        cond_steps: int | None = None,
    ) -> torch.Tensor:
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        t = t.clip(self.t_eps, 1.0 - self.t_eps)
        expand_shape = (-1,) + (x1.ndim - 1) * (1,)
        t_view = t.reshape(expand_shape)
        eps = torch.randn_like(x1) * self.noise_scale
        if cond_steps is not None:
            if not (1 <= cond_steps <= x1.shape[1]):
                raise ValueError(
                    f"cond_steps must be in [1, T], got cond_steps={cond_steps}, T={x1.shape[1]}"
                )
            eps[:, :cond_steps] = x1[:, :cond_steps]
        x_t = t_view * x1 + (1.0 - t_view) * eps
        if cond_steps is not None:
            x_t[:, :cond_steps] = x1[:, :cond_steps]
        # sample z ~ p(z|x1)
        mu, logvar = self.encoder(x1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        predictions = self.model(x_t, t, cond=z)
        fm_loss =  compute_flow_matching_loss(
            self.loss_type,
            x1=x1,
            eps=eps,
            predictions=predictions,
            cond_steps=cond_steps,
        )
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = 0.2 * kl.mean()
        return fm_loss + kl_loss, fm_loss.detach(), kl_loss.detach()
    
    @torch.inference_mode()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int,
    ) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        z = torch.randn((num_samples, self.latent_dim), device=device)
        x_t = torch.randn((num_samples,) + self.sample_shape, device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((num_samples,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, cond=z)
            x_t = x_t + v_hat * dt
        return x_t

    @torch.inference_mode()
    def sample_cond_prefix(
        self,
        cond_prefix: torch.Tensor,
        device: torch.device,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Sample a full window ``(N, T, D)`` with the first ``k`` timesteps fixed to
        ``cond_prefix`` (normalized space). Uses a single latent ``z ~ N(0, I)`` for the ODE,
        matching unconditional sampling semantics.
        """
        dtype = next(self.model.parameters()).dtype
        k = cond_prefix.shape[1]
        n = cond_prefix.shape[0]
        if cond_prefix.ndim != 3:
            raise ValueError(f"cond_prefix must be (N, k, D), got shape {tuple(cond_prefix.shape)}")
        seq_len, dim = self.sample_shape
        if cond_prefix.shape[2] != dim:
            raise ValueError(
                f"cond_prefix dim {cond_prefix.shape[2]} != sample_shape[1] {dim}"
            )
        if k > seq_len:
            raise ValueError(f"cond_steps k={k} exceeds seq_len={seq_len}")
        cond_prefix = cond_prefix.to(device=device, dtype=dtype)
        z = torch.randn((n, self.latent_dim), device=device, dtype=dtype)
        x_t = torch.randn((n,) + self.sample_shape, device=device, dtype=dtype)
        x_t = x_t * self.noise_scale
        x_t[:, :k] = cond_prefix
        ts = torch.linspace(self.t_eps, 1.0 - self.t_eps, num_steps, device=device, dtype=dtype)
        dt = (
            ts[1] - ts[0]
            if num_steps > 1
            else torch.tensor(1.0 - 2 * self.t_eps, device=device, dtype=dtype)
        )
        for t_scalar in ts:
            t = torch.full((n,), t_scalar.item(), device=device, dtype=dtype)
            _, v_hat, _ = self.model(x_t, t, cond=z)
            v_hat = v_hat.clone()
            v_hat[:, :k] = 0.0
            x_t = x_t + v_hat * dt
            x_t[:, :k] = cond_prefix
        return x_t


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)
