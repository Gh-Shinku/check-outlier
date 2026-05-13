import torch
import torch.nn as nn


class ActivationStatsHook:
    def __init__(self, model):
        self.stats = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.stats[name] = {
                    "max_abs": None,
                    "sum": None,
                    "sumsq": None,
                    "count": 0,
                }
                self.handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name):
        def hook(module, input, output):
            out = output.detach().cpu()
            flat = out.reshape(-1, out.shape[-1])
            s = self.stats[name]
            cur_max = flat.abs().max(dim=0).values
            if s["max_abs"] is None:
                s["max_abs"] = cur_max
            else:
                s["max_abs"] = torch.max(s["max_abs"], cur_max)
            if s["sum"] is None:
                s["sum"] = flat.sum(dim=0)
            else:
                s["sum"] += flat.sum(dim=0)
            if s["sumsq"] is None:
                s["sumsq"] = (flat**2).sum(dim=0)
            else:
                s["sumsq"] += (flat**2).sum(dim=0)
            s["count"] += flat.size(0)

        return hook

    def reset(self):
        for name in self.stats:
            self.stats[name] = {
                "max_abs": None,
                "sum": None,
                "sumsq": None,
                "count": 0,
            }

    def get_stats(self):
        return self.stats

    def remove(self):
        for h in self.handles:
            h.remove()
