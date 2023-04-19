"""
Sample script to train an ExponentiatedGradient model.
"""

from tableshift import get_dataset
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config

dset = get_dataset("_debug")

X_tr, y_tr, _, d_tr = dset.get_pandas(split="train")

config = get_default_config("expgrad", dset)
estimator = get_estimator("expgrad", **config)

estimator.fit(X_tr, y_tr, d=d_tr)

for split in ("id_test", "ood_test"):

    X_te, _, _, _ = dset.get_pandas(split=split)

    y_hat_te = estimator.predict(X_te)
    metrics = dset.evaluate_predictions(y_hat_te, split=split)
    print(f"metrics on split {split}:")
    for k, v in metrics.items():
        print(f"\t{k:<40}:{v:.3f}")
