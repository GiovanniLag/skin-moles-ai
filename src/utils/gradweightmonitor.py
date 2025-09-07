from typing import Iterable, Optional, Any, Mapping
import os
import time
import math
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

Batch = Any  # can be a Tensor, tuple/list/dict of Tensors, etc.

def _is_finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def _cpu_detach(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()

def _limit_batch_items(x: Batch, max_items: int) -> Batch:
    """Keep only the first max_items along batch dimension if possible."""
    if isinstance(x, torch.Tensor):
        if x.ndim >= 1 and x.size(0) > max_items:
            return x[:max_items]
        return x
    elif isinstance(x, Mapping):
        return {k: _limit_batch_items(v, max_items) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        t = type(x)
        return t(_limit_batch_items(v, max_items) for v in x)
    else:
        return x  # leave unknown types as-is

def _to_cpu_detached(x: Batch) -> Batch:
    if isinstance(x, torch.Tensor):
        return _cpu_detach(x)
    elif isinstance(x, Mapping):
        return {k: _to_cpu_detached(v) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        t = type(x)
        return t(_to_cpu_detached(v) for v in x)
    else:
        return x

class GradWeightMonitor(Callback):
    """
    Monitors gradients and weights; on NaN/Inf:
      • saves current batch (+ optional model/optimizer/scaler states)
      • requests training stop
      • logs which parameters were problematic

    Also logs grad/weight norms and optional histograms.
    """

    def __init__(
        self,
        log_hist_every_n_steps: int = 0,        # 0 = disable histograms
        grad_norm_p: float = 2.0,
        param_groups: Optional[Iterable[str]] = None,  # substrings to filter parameter names
        dump_dir: str = "nan_dumps",
        max_items_per_tensor: int = 8,          # limit batch dump size
        save_states: bool = True,               # also save model/optimizer/gradscaler
        stop_on_nonfinite: bool = True,
    ):
        super().__init__()
        self.log_hist_every_n_steps = log_hist_every_n_steps
        self.grad_norm_p = grad_norm_p
        self.param_groups = tuple(param_groups) if param_groups else None
        self.dump_dir = dump_dir
        self.max_items_per_tensor = max_items_per_tensor
        self.save_states = save_states
        self.stop_on_nonfinite = stop_on_nonfinite

        self._last_batch: Optional[Batch] = None
        self._stop_requested_once = False

    # ---------- utilities

    @staticmethod
    def _match(name: str, groups: Optional[Iterable[str]]) -> bool:
        return True if not groups else any(g in name for g in groups)

    def _global_grad_norm(self, named_params):
        parts = []
        device = None
        for name, p in named_params:
            if p.grad is None or not self._match(name, self.param_groups):
                continue
            g = p.grad
            if device is None and isinstance(g, torch.Tensor):
                device = g.device
            if isinstance(g, torch.Tensor) and torch.isfinite(g).all():
                parts.append(g.norm(p=2))
        if not parts:
            return torch.tensor(0.0, device=device or torch.device("cpu"))
        return torch.linalg.vector_norm(torch.stack(parts), ord=self.grad_norm_p)

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _dump_batch_and_states(
        self,
        pl_module: pl.LightningModule,
        trainer: pl.Trainer,
        tag: str,
        batch: Optional[Batch] = None,
    ) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run = f"step{trainer.global_step}_ep{trainer.current_epoch}_{tag}_{ts}"
        outdir = os.path.join(self.dump_dir, run)
        self._ensure_dir(outdir)

        # Batch
        b = batch if batch is not None else self._last_batch
        if b is not None:
            limited = _limit_batch_items(b, self.max_items_per_tensor)
            tosave = _to_cpu_detached(limited)
            torch.save(tosave, os.path.join(outdir, "batch.pt"))

        if self.save_states:
            # Model
            torch.save(pl_module.state_dict(), os.path.join(outdir, "model_state.pt"))
            # Optimizers (support multiple)
            opt_states = [opt.state_dict() for opt in trainer.optimizers] if hasattr(trainer, "optimizers") else []
            torch.save(opt_states, os.path.join(outdir, "optim_states.pt"))
            # GradScaler (AMP)
            scaler = getattr(trainer, "scaler", None) or getattr(trainer.strategy, "scaler", None)
            if scaler is not None:
                try:
                    torch.save(scaler.state_dict(), os.path.join(outdir, "gradscaler_state.pt"))
                except Exception:
                    pass

        # Quick text log
        with open(os.path.join(outdir, "info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Detected non-finite values at global_step={trainer.global_step}, epoch={trainer.current_epoch}\n")
            f.write(f"Tag: {tag}\n")

        pl_module.print(f"[GradWeightMonitor] Dumped debug artifacts to: {outdir}")
        return outdir

    def _request_stop(self, trainer: pl.Trainer, reason: str, pl_module: pl.LightningModule):
        if self._stop_requested_once:
            return
        self._stop_requested_once = True
        pl_module.print(f"[GradWeightMonitor] STOP requested: {reason}")
        # Graceful stop (Lightning 2.x); if unavailable, raising RuntimeError also halts.
        try:
            trainer.request_stop()
        except Exception:
            raise RuntimeError(f"Training stopped due to: {reason}")

    def _log_histograms(self, pl_module: pl.LightningModule, named_params, kind: str):
        if self.log_hist_every_n_steps <= 0:
            return
        logger = pl_module.logger
        if logger is None:
            return
        tb_add_hist = getattr(getattr(logger, "experiment", None), "add_histogram", None)
        wb_log = None
        try:
            import wandb
            wb_log = getattr(getattr(logger, "experiment", None), "log", None)
        except Exception:
            wb_log = None

        step = pl_module.global_step
        for name, p in named_params:
            if not self._match(name, self.param_groups):
                continue
            t = p.grad if kind == "grad" else p.data
            if t is None:
                continue
            t = t.detach().float().view(-1).cpu()
            if t.numel() == 0 or not torch.isfinite(t).any():
                continue
            if tb_add_hist is not None:
                tb_add_hist(f"{kind}_hist/{name}", t, global_step=step)
            if wb_log is not None:
                try:
                    import wandb
                    wb_log({f"{kind}_hist/{name}": wandb.Histogram(t.numpy())}, step=step)
                except Exception:
                    pass

    # ---------- Lightning hooks

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Batch, batch_idx: int):
        # Keep a reference to the current batch for dumping if needed
        self._last_batch = batch

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        """Called after backward; gradients are AMP-unscaled here."""
        named_params = list(pl_module.named_parameters())

        # Per-parameter checks and max grad-norm
        bad = []
        max_gn = 0.0
        for name, p in named_params:
            if p.grad is None or not self._match(name, self.param_groups):
                continue
            g = p.grad
            if not isinstance(g, torch.Tensor) or not torch.isfinite(g).all():
                bad.append(name)
            else:
                # per-tensor L2 norm
                n = g.norm(p=2).item()
                if math.isfinite(n):
                    max_gn = max(max_gn, n)

        global_norm = self._global_grad_norm(named_params).item() if named_params else 0.0

        pl_module.log("grad_norm/global", float(global_norm), on_step=True)
        pl_module.log("grad_norm/max_tensor", float(max_gn), on_step=True)
        pl_module.log("grad/has_nonfinite", float(len(bad) > 0), on_step=True, prog_bar=True)

        if bad:
            pl_module.print(f"[GradWeightMonitor] Non-finite GRADIENTS in: {bad[:8]}{'…' if len(bad)>8 else ''}")
            # Dump & stop immediately (before optimizer updates corrupt weights)
            self._dump_batch_and_states(pl_module, trainer, tag="nonfinite_grads")
            if self.stop_on_nonfinite:
                self._request_stop(trainer, "non-finite gradients", pl_module)

        # Optional hist logging
        if self.log_hist_every_n_steps and (trainer.global_step % self.log_hist_every_n_steps == 0):
            self._log_histograms(pl_module, named_params, kind="grad")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int
    ) -> None:
        """Check weights after the optimizer step."""
        named_params = list(pl_module.named_parameters())
        bad_w = []
        max_wn = 0.0
        for name, p in named_params:
            if not self._match(name, self.param_groups):
                continue
            w = p.data
            if not isinstance(w, torch.Tensor) or not torch.isfinite(w).all():
                bad_w.append(name)
            else:
                n = w.norm(p=2).item()
                if math.isfinite(n):
                    max_wn = max(max_wn, n)

        pl_module.log("weight_norm/max_tensor", float(max_wn), on_step=True)
        pl_module.log("weight/has_nonfinite", float(len(bad_w) > 0), on_step=True, prog_bar=True)

        if bad_w:
            pl_module.print(f"[GradWeightMonitor] Non-finite WEIGHTS in: {bad_w[:8]}{'…' if len(bad_w)>8 else ''}")
            # Dump & stop (we have direct access to batch here too)
            self._dump_batch_and_states(pl_module, trainer, tag="nonfinite_weights", batch=batch)
            if self.stop_on_nonfinite:
                self._request_stop(trainer, "non-finite weights", pl_module)

        if self.log_hist_every_n_steps and (trainer.global_step % self.log_hist_every_n_steps == 0):
            self._log_histograms(pl_module, named_params, kind="weight")
