import torch

def summarize(args, dataset, source_model):
    with torch.no_grad():
        source_model.eval()
        source_model.to(args.device)
        source_model_probs = None

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
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov