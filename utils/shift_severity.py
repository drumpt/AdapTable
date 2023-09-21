import numpy as np


def kl_divergence(h1, h2):
    """Calculate the KL divergence between two histograms."""
    h1_rel_freq = h1 / np.sum(h1)
    h2_rel_freq = h2 / np.sum(h2)
    nonzero_idx = (h1_rel_freq != 0) & (h2_rel_freq != 0)  # Avoid division by zero
    return max(np.sum(h1_rel_freq[nonzero_idx] * np.log(h1_rel_freq[nonzero_idx] / h2_rel_freq[nonzero_idx])), 0)


def js_distance(h1, h2):
    """Calculate the Jensen-Shannon distance between two histograms."""
    h1_rel_freq = h1 / (np.sum(h1) + np.finfo(float).eps)
    h2_rel_freq = h2 / (np.sum(h2) + np.finfo(float).eps)

    m = (h1_rel_freq + h2_rel_freq) / 2

    nonzero_idx1 = (h1_rel_freq != 0) & (m != 0)
    nonzero_idx2 = (h2_rel_freq != 0) & (m != 0)

    kl_div1 = np.sum(h1_rel_freq[nonzero_idx1] * np.log((h1_rel_freq[nonzero_idx1] + np.finfo(float).eps) /
                                                        (m[nonzero_idx1] + np.finfo(float).eps)))

    kl_div2 = np.sum(h2_rel_freq[nonzero_idx2] * np.log((h2_rel_freq[nonzero_idx2] + np.finfo(float).eps) /
                                                        (m[nonzero_idx2] + np.finfo(float).eps)))

    js_div = (kl_div1 + kl_div2) / 2
    return np.sqrt(js_div)  # Jensen-Shannon distance


def calculate_columnwise_kl_divergence(train_data, test_data, bins=1000):
    """Calculate the KL divergence for each pair of corresponding columns in two datasets."""
    num_columns = train_data.shape[1]
    kl_divergences = np.zeros(num_columns)

    for i in range(num_columns):
        # set range of the bins for both datasets
        min = np.min([np.min(train_data[:, i]), np.min(test_data[:, i])])
        max = np.max([np.max(train_data[:, i]), np.max(test_data[:, i])])

        train_hist, bin_edges = np.histogram(train_data[:, i], bins=bins, density=True, range=(min, max))
        test_hist, _ = np.histogram(test_data[:, i], bins=bin_edges, density=True, range=(min, max))

        kl_divergences[i] = js_distance(train_hist, test_hist)

    return kl_divergences