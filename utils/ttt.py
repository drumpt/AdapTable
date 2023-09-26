import torch
import numpy as np


def summarize(args, dataset, source_model):
    with torch.no_grad():
        source_model.eval()
        source_model.to(args.device)
        stacked_features = []
        for train_x, train_mask_y, train_y in dataset.test_loader:
            # cast all to device
            train_x, train_mask_y, train_y = train_x.to(args.device), train_mask_y.to(args.device), train_y.to(args.device)
            batched_feature = source_model.get_feature(train_x)
            stacked_features.append(batched_feature)

        stacked_features = torch.cat(stacked_features, dim=0).to(args.device)

        mean = stacked_features.mean(0)
        sigma = covariance(stacked_features)

        return mean, sigma

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    assert features.size(0) > 1, "TODO: single sample covariance"

    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss