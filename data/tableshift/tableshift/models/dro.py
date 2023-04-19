from typing import Optional, Mapping, Dict, Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tableshift.models.compat import SklearnStylePytorchModel
from tableshift.models.rtdl import MLPModel
from tableshift.models.torchutils import unpack_batch, apply_model


class GroupDROModel(MLPModel, SklearnStylePytorchModel):
    def __init__(self, group_weights_step_size: float, n_groups: int, **kwargs):
        MLPModel.__init__(self, **kwargs)

        assert n_groups > 0, "require nonzero n_groups."
        self.group_weights_step_size = torch.Tensor([group_weights_step_size])
        # initialize adversarial weights
        self.group_weights = torch.nn.Parameter(
            torch.full([n_groups], 1. / n_groups))

    def to(self, device):
        super().to(device)
        for attr in ("group_weights_step_size", "group_weights"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Mapping[str, DataLoader]] = None,
                    # Terminate after this many steps if reached before end
                    # of epoch.
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]

        for iteration, batch in tqdm(enumerate(train_loader),
                                     desc="groupdro:train"):
            x_batch, y_batch, _, d_batch = unpack_batch(batch)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            d_batch = d_batch.float().to(device)
            self.train()
            self.optimizer.zero_grad()
            outputs = apply_model(self, x_batch)
            loss = loss_fn(outputs.squeeze(1), y_batch, d_batch,
                           self.group_weights,
                           self.group_weights_step_size)

            loss.backward()
            self.optimizer.step()
