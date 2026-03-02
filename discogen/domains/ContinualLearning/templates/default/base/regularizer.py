"""EWC-style regularizer."""

from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn


@dataclass
class RegularizerConfig:
    importance_estimation_batches: int = 10
    importance_decay: float = 1.0
    max_val_batches: int = 50
    eps: float = 1e-8


class Regularizer:
    def __init__(self, cfg: RegularizerConfig, seed: int):
        self.cfg = cfg
        self.old_params: Dict[str, torch.Tensor] = {}
        self.importance: Dict[str, torch.Tensor] = {}

    def on_task_start(self, task_id: int, model: nn.Module) -> None:
        # No-op; state is updated at task end
        pass

    def compute_penalty(self, model: nn.Module, step: int) -> torch.Tensor:
        if not self.old_params:
            return torch.zeros((), device=next(model.parameters()).device)
        loss = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.old_params:
                imp = self.importance[name].to(p.device)
                old_p = self.old_params[name].to(p.device)
                if p.shape != old_p.shape:
                    sliced_p = p[:old_p.shape[0]]
                    delta = sliced_p - old_p
                else:
                    delta = p - old_p

                loss = loss + (imp * (delta * delta)).sum()
        return loss

    def on_task_end(self, task_id: int, model: nn.Module, val_loader) -> None:
        # Decay previous importance
        for k in list(self.importance.keys()):
            self.importance[k] = self.importance[k] * self.cfg.importance_decay

        # Estimate diagonal Fisher from validation subset using CE loss
        model.train()
        steps = 0
        for batch in val_loader:
            x, y = batch
            x = x.to(next(model.parameters()).device)
            y = y.to(x.device)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g2 = (p.grad.detach() ** 2)
                if name not in self.importance:
                    self.importance[name] = g2
                else:
                    current_imp = self.importance[name]
                    if current_imp.shape != g2.shape:
                        new_imp = torch.zeros_like(g2)

                        if current_imp.dim() == 1:
                            new_imp[:current_imp.shape[0]] = current_imp
                        else:
                            new_imp[:current_imp.shape[0], :] = current_imp

                        self.importance[name] = new_imp + g2
                    else:
                        self.importance[name] += g2
            steps += 1
            if steps >= min(self.cfg.importance_estimation_batches, self.cfg.max_val_batches):
                break

        # Average across steps and add epsilon
        if steps > 0:
            for k in self.importance:
                self.importance[k] = self.importance[k] / float(steps) + self.cfg.eps

        # Snapshot old params
        self.old_params = {name: p.detach().clone().cpu() for name, p in model.named_parameters() if p.requires_grad}

    def state_dict(self) -> Dict[str, Any]:
        return {"old_params": self.old_params, "importance": self.importance, "cfg": self.cfg.__dict__}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.old_params = state.get("old_params", {})
        self.importance = state.get("importance", {})
