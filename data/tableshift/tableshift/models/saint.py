import copy
from typing import Optional, Callable, Any, Dict

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader

from tableshift.models.compat import SklearnStylePytorchModel, OPTIMIZER_ARGS
from tableshift.models.training import train_epoch
from tableshift.third_party.saint.models import SAINT
from tableshift.third_party.saint.augmentations import embed_data_mask


class SaintModel(SAINT, SklearnStylePytorchModel):
    """Adapted version of SAINT model."""

    def __init__(self, **hparams):
        self.config = copy.deepcopy(hparams)

        # Remove hparams that are not taken by the SAINT constructor.
        for k in OPTIMIZER_ARGS:
            hparams.pop(k)

        super().__init__(**hparams)
        self._init_optimizer()

    @torch.no_grad()
    def predict_proba(self, X):
        prediction = self.predict(X)
        return scipy.special.expit(prediction)

    def train_epoch(self,
                    train_loaders: Dict[Any, DataLoader],
                    loss_fn: Callable,
                    device: str,
                    uda_loader: Optional[DataLoader] = None,
                    eval_loaders: Optional[Dict[str, DataLoader]] = None,
                    max_examples_per_epoch: Optional[int] = None
                    ) -> float:
        """Run a single epoch of model training."""
        assert len(train_loaders.values()) == 1
        train_loader = list(train_loaders.values())[0]
        return train_epoch(self, self.optimizer, loss_fn, train_loader, device)

    def forward(self, x_cont, x_categ):
        """Overrides forward() with modifications.

        Reverses the inputs to SAINT.forward(), but also implements a complete
        forward pass (from inputs -> output probabilities) instead of requiring
        more complex logic outside the foward pass as in the original SAINT
        implementation (e.g. compare to
        https://github.com/somepago/saint/blob/main/train.py#L183 )
        """
        assert not x_categ and not self.num_categories, \
            "categorical SAINT not currently supported."

        cat_mask = torch.zeros(len(x_cont)).int().to(x_cont.device)
        con_mask = torch.ones_like(x_cont).int().to(x_cont.device)
        # We are converting the data to embeddings in the next step
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ=x_categ,
                                                     x_cont=x_cont,
                                                     cat_mask=cat_mask,
                                                     con_mask=con_mask,
                                                     model=self)
        reps = self.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to CLS token and
        # apply mlp on it in the next step to get the predictions.
        y_reps = reps[:, 0, :]

        y_outs = self.mlpfory(y_reps)
        return y_outs

    def predict_proba(self, X) -> np.ndarray:
        """sklearn-compatible probability prediction function."""
        raise
