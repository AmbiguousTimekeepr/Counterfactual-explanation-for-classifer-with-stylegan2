import copy
import torch


class EMA:
    """Exponential Moving Average utility for stabilizing generator weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update the EMA weights with the current model weights."""
        msd = model.state_dict()
        esd = self.model.state_dict()
        for key, value in msd.items():
            if value.dtype.is_floating_point:
                esd[key].data.mul_(self.decay).add_(value.data, alpha=1 - self.decay)
