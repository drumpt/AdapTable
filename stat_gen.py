import warnings

import fontTools.cffLib
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from data.dataset import *
from model.model import *
from utils.utils import *
from utils.sam import *
from utils.class_prototype import *

tsne = TSNE(learning_rate=200, perplexity=50, n_iter=3000, early_exaggeration=12, init='pca', angle=0.5)


def permute_columns(tensor):
    batch_size, num_columns = tensor.size()
    permuted_tensor = torch.empty_like(tensor)

    for i in range(batch_size):
        permuted_indices = torch.randperm(num_columns)
        permuted_tensor[i] = tensor[i, permuted_indices]

    return permuted_tensor

def plot_mean_std_column(args, dataset):
    # print statistics of the dataset, per column
    np_train_x = dataset.train_x
    np_train_y = dataset.train_y

    np_test_x = dataset.test_x
    np_test_y = dataset.test_y

    per_column_train_mean = np.mean(np_train_x, axis=0)
    per_column_train_std = np.std(np_train_x, axis=0)

    per_column_test_mean = np.mean(np_test_x, axis=0)
    per_column_test_std = np.std(np_test_x, axis=0)

    print('per_column_train_mean : ', per_column_train_mean)
    print('per_column_train_std : ', per_column_train_std)

    print('per_colum_test_mean : ', per_column_test_mean)
    print('per_colum_test_std : ', per_column_test_std)

    columns = np.arange(len(per_column_train_mean))
    num_columns = len(columns)
    cols_per_subplot = 40

    # Calculate the number of subplots required
    num_subplots = -(-num_columns // cols_per_subplot)  # Ceiling division

    plt.figure(figsize=(15, 6 * num_subplots))

    for i in range(num_subplots):
        start_idx = i * cols_per_subplot
        end_idx = min(start_idx + cols_per_subplot, num_columns)

        plt.subplot(num_subplots, 1, i + 1)

        plt.errorbar(columns[start_idx:end_idx] - 0.3,
                     per_column_train_mean[start_idx:end_idx],
                     yerr=per_column_train_std[start_idx:end_idx],
                     fmt='o', color='b', ecolor='b', elinewidth=2,
                     capsize=5, label='Train', alpha=0.6)

        plt.errorbar(columns[start_idx:end_idx] + 0.3,
                     per_column_test_mean[start_idx:end_idx],
                     yerr=per_column_test_std[start_idx:end_idx],
                     fmt='o', color='r', ecolor='r', elinewidth=2,
                     capsize=5, label='Test', alpha=0.6)

        if i == 0:  # Only add legend for the first subplot to avoid redundancy
            plt.title(f'Dataset: {args.dataset} Per-Column Mean with STD')
            plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_output(args, dataset, source_model, test_model):
    torch_train_x = torch.from_numpy(dataset.train_x).float().to(args.device)
    torch_train_y = torch.from_numpy(dataset.train_y).float().to(args.device)

    torch_test_x = torch.from_numpy(dataset.test_x).float().to(args.device)
    torch_test_y = torch.from_numpy(dataset.test_y).float().to(args.device)


    for idx, model in enumerate([source_model, test_model]):
        mask = torch.ones_like(torch_train_x[0])
        masked_train_x = torch_train_x * mask
        masked_test_x = torch_test_x * mask


        model = model.to(args.device)
        train_feature = (model.get_feature(masked_train_x)).cpu().detach().numpy()
        test_feature = (model.get_feature(masked_test_x)).cpu().detach().numpy()

        train_y = np.argmax(dataset.train_y, axis=1)
        test_y = np.argmax(dataset.test_y, axis=1)

        tsne = TSNE(n_components=2, random_state=0)
        train_feature = tsne.fit_transform(train_feature)

        tsne = TSNE(n_components=2, random_state=0)
        test_feature = tsne.fit_transform(test_feature)

        plt.figure(figsize=(10, 20))
        plt.subplot(2, 1, 1)
        plt.scatter(train_feature[:, 0], train_feature[:, 1], c=train_y, cmap='rainbow')
        plt.title('train features')

        plt.subplot(2, 1, 2)
        plt.scatter(test_feature[:, 0], test_feature[:, 1], c=test_y, cmap='rainbow')
        plt.title('test features')
        plt.show()

    plt.figure(figsize=(15, 6))
    from utils.utils import get_shap_values
    src_shap_importance, _ = get_shap_values(dataset, source_model,
                                                      torch.tensor(dataset.train_x).float().to(args.device), args)
    src_shap_importance_test, _ = get_shap_values(dataset, source_model,
                                             torch.tensor(dataset.train_x).float().to(args.device), args)

    test_shap_importance, _ = get_shap_values(dataset, test_model,
                                              torch.tensor(dataset.test_x).float().to(args.device),
                                              args)
    test_shap_importance_test, _ = get_shap_values(dataset, test_model,
                                              torch.tensor(dataset.train_x).float().to(args.device),
                                              args)

    src_shap_importance = np.mean(np.abs(src_shap_importance), axis=0)
    src_shap_importance_test = np.mean(np.abs(src_shap_importance_test), axis=0)
    test_shap_importance_test = np.mean(np.abs(test_shap_importance_test), axis=0)
    test_shap_importance = np.mean(np.abs(test_shap_importance), axis=0)

    plt.subplot(4, 1, 1)
    plt.bar(range(len(src_shap_importance)), src_shap_importance)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Source model importance upon its training data')
    plt.xticks(range(len(src_shap_importance)))

    plt.subplot(4, 1, 2)
    plt.bar(range(len(src_shap_importance_test)), src_shap_importance_test)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Source model importance upon its test data')
    plt.xticks(range(len(src_shap_importance_test)))

    plt.subplot(4, 1, 3)
    plt.bar(range(len(test_shap_importance)), test_shap_importance)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Test model importance upon its training data')
    plt.xticks(range(len(test_shap_importance)))

    plt.subplot(4, 1, 4)
    plt.bar(range(len(test_shap_importance_test)), test_shap_importance_test)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Test model importance upon its test data')
    plt.xticks(range(len(test_shap_importance_test)))
    plt.show()

def plot_learned_bias(args, model, dataset):
    model = model.to(args.device)
    bias = torch.zeros_like(torch.tensor(dataset.train_x[0]).float().to(args.device))
    bias = bias.to(args.device)
    bias.requires_grad_(True)
    optimizer = torch.optim.Adam([bias], lr=0.1)
    model.requires_grad_(False)

    for epoch in range(50):
        optimizer.zero_grad()
        biased_output = model(torch.tensor(dataset.train_x).float().to(args.device) + bias)
        loss = F.cross_entropy(biased_output, torch.argmax(torch.tensor(dataset.train_y).float().to(args.device), dim=-1))
        loss.backward()
        optimizer.step()
        print(loss.item())
    bias = bias.cpu().detach().numpy()


    bias_test = torch.zeros_like(torch.tensor(dataset.train_x[0]).float().to(args.device))
    bias_test = bias_test.to(args.device)
    bias_test.requires_grad_(True)
    optimizer = torch.optim.Adam([bias_test], lr=0.1)
    for epoch in range(50):
        optimizer.zero_grad()
        biased_output = model(torch.tensor(dataset.test_x).float().to(args.device) + bias_test)
        loss = F.cross_entropy(biased_output, torch.argmax(torch.tensor(dataset.test_y).float().to(args.device), dim=-1))
        loss.backward()
        optimizer.step()
        print(loss.item())
    bias_test = bias_test.cpu().detach().numpy()

    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.bar(range(len(bias)), bias)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Learned bias')
    plt.xticks(range(len(bias)))

    plt.subplot(3, 1, 2)
    plt.bar(range(len(bias_test)), bias_test)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Learned bias test')
    plt.xticks(range(len(bias_test)))

    plt.subplot(3, 1, 3)
    plt.bar(range(len(bias_test)), np.clip(np.abs(bias_test - bias) / np.abs(bias), 0, 50))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Difference between learned bias and learned bias test')
    plt.xticks(range(len(bias_test)))
    plt.show()


def plot_recon_feature(args, dataset, model):
    torch_train_x = torch.from_numpy(dataset.train_x).float().to(args.device)
    torch_train_y = torch.from_numpy(dataset.train_y).float().to(args.device)

    torch_test_x = torch.from_numpy(dataset.test_x).float().to(args.device)
    torch_test_y = torch.from_numpy(dataset.test_y).float().to(args.device)

    from utils.utils import get_shap_values
    shap_importance, shap_explainer =  get_shap_values(dataset, model, torch.tensor(dataset.train_x).float().to(args.device), args)

    print(shap_importance)

    idx_unimportant = np.argsort(shap_importance, axis=-1)[:int(len(shap_importance) * 0.25)]
    mask = torch.zeros_like(torch_train_x[0]).to(args.device)
    mask[idx_unimportant] = 1

    # dropout = nn.Dropout(p=0.8)
    torch_train_x = torch_train_x * mask
    torch_test_x = torch_test_x * mask

    model = model.to(args.device)
    train_feature = model.get_feature(model.get_recon_out(torch_train_x)).cpu().detach().numpy()
    test_feature = model.get_feature(model.get_recon_out(torch_test_x)).cpu().detach().numpy()

    train_y = np.argmax(dataset.train_y, axis=1)
    test_y = np.argmax(dataset.test_y, axis=1)

    all_features = np.vstack([train_feature, test_feature])
    all_labels = np.hstack([train_y, test_y])

    tsne_results = tsne.fit_transform(all_features)

    # Split the t-SNE results back into train and test datasets
    train_tsne_results = tsne_results[:len(train_feature)]
    test_tsne_results = tsne_results[len(train_feature):]

    plt.figure(figsize=(12, 6))

    # Calculate the global limits for x and y to ensure same width and height for both subplots
    x_min = min(train_tsne_results[:, 0].min(), test_tsne_results[:, 0].min())
    x_max = max(train_tsne_results[:, 0].max(), test_tsne_results[:, 0].max())

    y_min = min(train_tsne_results[:, 1].min(), test_tsne_results[:, 1].min())
    y_max = max(train_tsne_results[:, 1].max(), test_tsne_results[:, 1].max())

    plt.subplot(1, 2, 1)
    for label in np.unique(train_y):
        idx = train_y == label
        plt.scatter(train_tsne_results[idx, 0], train_tsne_results[idx, 1], label=f"Label {label}")
    plt.title("Train Features")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.subplot(1, 2, 2)
    for label in np.unique(test_y):
        idx = test_y == label
        plt.scatter(test_tsne_results[idx, 0], test_tsne_results[idx, 1], label=f"Label {label}")
    plt.title("Test Features")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.tight_layout()  # Adjusts subplots for better layout
    plt.show()

def plot_acc_over_relative_entropy(args, dataset, model):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import softmax

    def compute_relative_entropy(predictions, num_classes, temperature=1.0):
        softmax_output = softmax(predictions / temperature, axis=1)
        entropy = -np.sum(softmax_output * np.log(softmax_output + 1e-10), axis=1)  # Add a small constant to avoid log(0)
        relative_entropy = entropy / np.log(num_classes)
        return relative_entropy

    def plot_accuracy_vs_binned_entropy(predictions, true_labels, bin_count, temperature=1.0):
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.special

        def compute_relative_entropy(predictions, num_classes):
            entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
            relative_entropy = entropy / np.log(num_classes)
            return relative_entropy

        def apply_temperature_scaling(logits, temperature):
            return logits / temperature

        def plot_accuracies_and_label_distribution(relative_entropies, accuracies, labels, bin_count):
            unique_labels = np.unique(labels)
            label_colors = [f'C{i}' for i in range(len(unique_labels))]

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Create secondary Y-axis for label distribution
            ax2 = ax1.twinx()
            ax2.set_ylabel('Label Distribution', color='tab:blue')

            # quantiles = np.linspace(0, 1, bin_count + 1)
            # quantile_bins = np.quantile(relative_entropies, quantiles)
            bin_edges = np.quantile(relative_entropies, q=np.linspace(0, 1, bin_count + 1))

            # Get the bin index for each value in relative_entropies
            bin_indices = np.digitize(relative_entropies, bin_edges) - 1
            # Ensure bin_indices are in the range [0, bin_count-1]
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Plot accuracies
            ax1.set_xlabel('Relative Entropy Bins')
            ax1.set_ylabel('Accuracy', color='tab:red')
            ax1.plot(bin_midpoints, accuracies, color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            # Create stacked bars for each label
            bottoms = np.zeros(bin_count)
            for idx, label in enumerate(unique_labels):
                label_mask = labels == label
                label_distribution = [
                    np.sum(label_mask[bin_indices == bin_idx]) / np.sum(bin_indices == bin_idx)
                    for bin_idx in range(bin_count)]
                ax2.bar(bin_midpoints, label_distribution, bottom=bottoms, color=label_colors[idx], alpha=0.3,
                        width=0.01, label=f"Label {label}")
                bottoms = bottoms + np.array(label_distribution)

            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))

            fig.tight_layout()  # Otherwise the right y-label is slightly clipped
            plt.show()

        def process_bins(predictions, labels, temperatures, bin_count):
            predictions_woscaling = predictions
            predictions = softmax(predictions, axis=1)
            num_classes = predictions.shape[1]
            logits = predictions_woscaling

            # 1. Compute relative entropies without temperature scaling
            relative_entropies = compute_relative_entropy(predictions, num_classes)

            # 2. Bin them regarding their relative entropies
            bins = np.quantile(relative_entropies, q=np.linspace(0, 1, bin_count + 1))
            bin_indices = np.digitize(relative_entropies, bins) - 1

            # 3. Re-calculate their relative entropies, per bin, with temperature scaling
            for i in range(bin_count):
                logits[bin_indices == i] = apply_temperature_scaling(logits[bin_indices == i], temperatures[i])
            new_predictions = scipy.special.softmax(logits, axis=1)
            new_relative_entropies = compute_relative_entropy(new_predictions, num_classes)

            # 4. Compute each bin's accuracies
            accuracies = [np.mean(labels[bin_indices == i] == np.argmax(new_predictions[bin_indices == i], axis=1)) for
                          i in
                          range(bin_count)]

            # 5. Plot accuracies and label distribution
            plot_accuracies_and_label_distribution(new_relative_entropies, accuracies, labels, bin_count)

        def generate_temperatures(n, k=10, a=0.5):
            x = np.linspace(0, 1, n)
            temperatures = 1 / (1 + np.exp(-k * (x - a)))
            return temperatures

        temperatures = generate_temperatures(bin_count)
        print('temperatures', temperatures)
        process_bins(predictions, true_labels, temperatures, bin_count)

    def plot_accuracy_vs_relative_entropy(predictions, true_labels, bin_count, temperature=1.0):
        # Number of classes is assumed to be the number of columns in the predictions
        num_classes = predictions.shape[1]

        # 1. Compute relative entropy
        relative_entropies = compute_relative_entropy(predictions, num_classes, temperature=temperature)
        bins = np.quantile(relative_entropies, np.linspace(0, 1, bin_count + 1))
        bins[0] = -np.inf  # ensure that all values are included in the bins
        bins[-1] = np.inf
        bin_indices = np.digitize(relative_entropies, bins) - 1

        # 3. Compute accuracy within each bin
        accuracies = []
        for i in range(bin_count):
            idx = bin_indices == i
            correct = np.sum(np.argmax(predictions[idx], axis=1) == true_labels[idx])
            total = np.sum(idx)
            accuracies.append(correct / total if total > 0 else 0)

        # 4. Plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, accuracies, '-o')
        plt.xlabel('Relative Entropy')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Relative Entropy temperature={}'.format(temperature))
        plt.grid(True)
        plt.show()

    # Simulate some data
    num_samples_train = len(dataset.train_y)
    num_samples_test = len(dataset.test_y)

    model = model.to(args.device)
    torch_train_x = torch.tensor(dataset.train_x).float().to(args.device)
    torch_test_x = torch.tensor(dataset.test_x).float().to(args.device)
    predictions_train = model(torch_train_x).cpu().detach().numpy()
    predictions_test = model(torch_test_x).cpu().detach().numpy()

    num_classes = dataset.train_y.shape[1]
    true_labels_train = np.argmax(dataset.train_y, axis=1)
    true_labels_test = np.argmax(dataset.test_y, axis=1)

    # Plot for training data
    plot_accuracy_vs_relative_entropy(predictions_test, true_labels_test, bin_count=10, temperature=0.8)
    plot_accuracy_vs_binned_entropy(predictions_test, true_labels_test, bin_count=10, temperature=0.8)

    # Plot for test data
    # plot_accuracy_vs_relative_entropy(predictions_test, true_labels_test, bin_count=10)

def plot_stunt_tasks(args, dataset, model):
    from utils.stunt_prototype import StuntPrototype

    train_x, train_y = torch.tensor(dataset.train_x).to(args.device), torch.tensor(dataset.train_y).to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).to(args.device), torch.tensor(dataset.test_y).to(args.device)

    emd_imputed_train_x = Dataset.get_imputed_data(train_x, dataset.train_x, data_type="numerical", imputation_method="zero")
    emd_imputed_test_x = Dataset.get_imputed_data(test_x, dataset.train_x, data_type="numerical", imputation_method="zero")
    model = model.to(args.device)

    for important_idx in range(train_x.shape[1]):

        train_prototype_x, train_prototype_y, train_prototype_target = StuntPrototype.construct_label_and_input(model, train_x, emd_imputed_train_x, important_idx, num_bins=10)
        test_prototype_x, test_prototype_y, test_prototype_target = StuntPrototype.construct_label_and_input(model, test_x, emd_imputed_test_x, important_idx, num_bins=10)

        train_feature = train_prototype_x.detach().cpu().numpy()
        test_feature = test_prototype_x.detach().cpu().numpy()

        train_prototype_target = train_prototype_target.detach().cpu().numpy()
        test_prototype_target = test_prototype_target.detach().cpu().numpy()

        all_features = np.vstack([train_feature, test_feature])
        all_labels = np.hstack([train_prototype_target, test_prototype_target])

        tsne_results = tsne.fit_transform(all_features)

        # Split the t-SNE results back into train and test datasets
        train_tsne_results = tsne_results[:len(train_feature)]
        test_tsne_results = tsne_results[len(train_feature):]

        plt.figure(figsize=(12, 6))

        # Calculate the global limits for x and y to ensure same width and height for both subplots
        x_min = min(train_tsne_results[:, 0].min(), test_tsne_results[:, 0].min())
        x_max = max(train_tsne_results[:, 0].max(), test_tsne_results[:, 0].max())

        y_min = min(train_tsne_results[:, 1].min(), test_tsne_results[:, 1].min())
        y_max = max(train_tsne_results[:, 1].max(), test_tsne_results[:, 1].max())

        plt.subplot(1, 2, 1)
        for label in np.unique(train_prototype_target):
            idx = train_prototype_target == label
            plt.scatter(train_tsne_results[idx, 0], train_tsne_results[idx, 1], label=f"Label {label}")
        plt.title("Train Features")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()

        plt.subplot(1, 2, 2)
        for label in np.unique(test_prototype_target):
            idx = test_prototype_target == label
            plt.scatter(test_tsne_results[idx, 0], test_tsne_results[idx, 1], label=f"Label {label}")
        plt.title("Test Features")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()

        plt.tight_layout()  # Adjusts subplots for better layout
        plt.show()


def plot_maskratio(args, dataset, model):
    from utils.stunt_prototype import StuntPrototype

    train_x, train_y = torch.tensor(dataset.train_x).to(args.device), torch.tensor(dataset.train_y).to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).to(args.device), torch.tensor(dataset.test_y).to(args.device)

    emd_imputed_train_x = Dataset.get_imputed_data(train_x, dataset.train_x, data_type="numerical", imputation_method="zero")
    emd_imputed_test_x = Dataset.get_imputed_data(test_x, dataset.train_x, data_type="numerical", imputation_method="zero")
    model = model.to(args.device)

    features_list = []
    labels_list = []
    shapely_values, _ = get_shap_values(dataset, model, train_x, args)
    shapely_values = np.sum(np.abs(shapely_values), axis=0)

    for idx, mask_ratio in enumerate([0, 0.8]):
        # random mask for train data
        mask_idx = np.argsort(shapely_values)[:int(train_x.shape[1] * mask_ratio)]
        train_mask = np.ones(train_x.shape[1])
        train_mask[mask_idx] = 0
        train_mask = torch.tensor(train_mask).float().to(args.device)
        # train_mask = torch.tensor(train_mask).float().to(args.device)
        emd_imputed_train_x = torch.tensor(emd_imputed_train_x).float().to(args.device)
        emd_imputed_test_x = torch.tensor(emd_imputed_test_x).float().to(args.device)
        x = test_x * train_mask + emd_imputed_test_x * (1 - train_mask)
        x = x.float()
        feature_x = model.get_feature(x)

        features_list.append(feature_x.detach().cpu().numpy())
        labels_list.append(np.array([idx for _ in range(feature_x.shape[0])]))

    tsne_list = []
    for i in range(len(features_list)):
        tsne = TSNE(n_components=2, random_state=0)
        tsne.fit_transform(features_list[i])
        tsne_list.append(tsne)


    tsne_results = np.vstack(tsne_list)
    all_labels = np.hstack(labels_list)

    x_min = tsne_results[:, 0].min()
    x_max = tsne_results[:, 0].max()

    y_min = tsne_results[:, 1].min()
    y_max = tsne_results[:, 1].max()

    for label in np.unique(all_labels):
        idx = all_labels == label
        mask_ratio_x = tsne_results[idx]
        mask_ratio_y = np.argmax(dataset.test_y, 1)
        for gt_label in np.unique(mask_ratio_y):
            idx_inner = mask_ratio_y == gt_label
            plt.scatter(mask_ratio_x[idx_inner, 0], mask_ratio_x[idx_inner, 1], label=f"Label {gt_label}")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title(f"Mask Ratio {label}")
        plt.legend()
        plt.show()
    # for label in np.unique(all_labels):
    #     idx = all_labels == label
    #     plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f"Label {label}")
    #     plt.xlim(x_min, x_max)
    #     plt.ylim(y_min, y_max)
    #     plt.title(f"Mask Ratio {label}")
    #     plt.show()

def pooled_avg_plot(args, model, dataset):
    model = model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).to(args.device), torch.tensor(dataset.train_y).to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).to(args.device), torch.tensor(dataset.test_y).to(args.device)

    train_x = train_x.float()
    test_x = test_x.float()

    for data in [(train_x, train_y), (test_x, test_y)]:
        x, y = data
        y = np.argmax(y.detach().cpu().numpy(), axis=-1)
        do_feature_list = []
        num_dropout = 1
        for idx in range(num_dropout):
            dropout = nn.Dropout(p=idx/num_dropout)
            do_feature = model.get_feature(dropout(x))
            do_feature_list.append(do_feature.detach().cpu().numpy())
        do_feature = np.average(do_feature_list, axis=0)

        tsne = TSNE(n_components=2, random_state=0)
        do_tsne_feature = tsne.fit_transform(do_feature)

        x_min = do_tsne_feature[:, 0].min()
        x_max = do_tsne_feature[:, 0].max()
        y_min = do_tsne_feature[:, 1].min()
        y_max = do_tsne_feature[:, 1].max()

        for label in range(dataset.train_y.shape[1]):
            idx = y == label
            plt.scatter(do_tsne_feature[idx, 0], do_tsne_feature[idx, 1], label=f"Label {label}")
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
        plt.legend()
        plt.show()

def plot_recon_loss_per_column_per_class(args, model, dataset):
    model = model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    dropout = nn.Dropout(p=0.75)


    with torch.no_grad():
        list_of_recon_loss = []
        for num_iter in range(1):
            recon_w_mask = model.get_recon_out(dropout(train_x))
            mse_loss_fn = nn.MSELoss(reduction="none")

            recon_loss_per_column = mse_loss_fn(recon_w_mask, train_x)
            recon_loss_per_column = recon_loss_per_column.detach().cpu().numpy()
            list_of_recon_loss = recon_loss_per_column

        list_of_recon_loss_test = []
        for num_iter in range(1):
            recon_w_mask = model.get_recon_out(dropout(test_x))
            mse_loss_fn = nn.MSELoss(reduction="none")

            recon_loss_per_column = mse_loss_fn(recon_w_mask, test_x)
            recon_loss_per_column = recon_loss_per_column.detach().cpu().numpy()
            list_of_recon_loss_test = recon_loss_per_column

        list_of_recon_loss = np.array(list_of_recon_loss)
        list_of_recon_loss_test = np.array(list_of_recon_loss_test)

        # idx list for each label
        idx_list = []
        for label in range(dataset.train_y.shape[1]):
            idx = train_y[:, label] == 1
            idx_list.append(idx.detach().cpu().numpy())

        idx_list_test = []
        for label in range(dataset.test_y.shape[1]):
            idx = test_y[:, label] == 1
            idx_list_test.append(idx.detach().cpu().numpy())

        num_plots = len(idx_list)
        plt.figure(figsize=(10, 5 * num_plots))
        for i, idx in enumerate(idx_list):
            per_class_per_column_recon = list_of_recon_loss[idx, :].mean(0)

            plt.subplot(num_plots, 1, i + 1)
            plt.bar(range(len(per_class_per_column_recon)), per_class_per_column_recon)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(' Train Recon loss of class {} per column'.format(i))
            plt.xticks(range(len(per_class_per_column_recon)))

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5 * num_plots))
        for i, idx in enumerate(idx_list_test):
            per_class_per_column_recon = list_of_recon_loss_test[idx, :].mean(0)

            plt.subplot(num_plots, 1, i + 1)
            plt.bar(range(len(per_class_per_column_recon)), per_class_per_column_recon)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(' Test Recon loss of class {} per column'.format(i))
            plt.xticks(range(len(per_class_per_column_recon)))

        plt.tight_layout()
        plt.show()


def plot_recon_loss_per_column(args, model, dataset):
    model = model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(args.device)
    dropout = nn.Dropout(p=0.75)

    list_of_recon_loss = []
    for num_iter in range(1):

        recon_w_mask = model.get_recon_out(dropout(train_x))
        mse_loss_fn = nn.MSELoss(reduction="none")

        recon_loss_per_column = mse_loss_fn(recon_w_mask, train_x).mean(0)
        recon_loss_per_column = recon_loss_per_column.detach().cpu().numpy()
        list_of_recon_loss.append(recon_loss_per_column)



    list_of_recon_loss_test = []
    for num_iter in range(1):
        recon_w_mask = model.get_recon_out(dropout(test_x))
        mse_loss_fn = nn.MSELoss(reduction="none")

        recon_loss_per_column = mse_loss_fn(recon_w_mask, test_x).mean(0)
        recon_loss_per_column = recon_loss_per_column.detach().cpu().numpy()
        list_of_recon_loss_test.append(recon_loss_per_column)

    model.train()
    model.requires_grad_(False)
    mask_vector = torch.ones_like(train_x[0]).float().to(args.device)
    mask_vector = mask_vector.requires_grad_(True)
    optimizer = torch.optim.Adam([mask_vector], lr=1e-1)

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
            #set range of the bins for both datasets
            min = np.min([np.min(train_data[:, i]), np.min(test_data[:, i])])
            max = np.max([np.max(train_data[:, i]), np.max(test_data[:, i])])

            train_hist, bin_edges = np.histogram(train_data[:, i], bins=bins, density=True, range=(min, max))
            test_hist, _ = np.histogram(test_data[:, i], bins=bin_edges, density=True, range=(min, max))

            kl_divergences[i] = js_distance(train_hist, test_hist)

        return kl_divergences

    # generate class prototypes
    class_prototypes = []
    features_train = model.get_feature_shallow(train_x)
    for label in range(dataset.train_y.shape[1]):
        idx = train_y[:, label] == 1
        class_prototypes.append(torch.mean(features_train[idx], 0))
    print(class_prototypes)
    print(f'current class is : {train_y[0]}')

    for num_iter in range(100):
        # mask_vector = torch.clamp(mask_vector, 0, 1)
        single_x = train_x[1]
        # recon_single_x = model.get_recon_out(single_x * mask_vector)
        shallow_single_x = model.get_feature_shallow(single_x * mask_vector)

        mse_loss_fn = nn.MSELoss(reduction="none")

        optimizer.zero_grad()
        mse_loss = mse_loss_fn(shallow_single_x, class_prototypes[1]).mean()
        # reg_loss = F.l1_loss(mask_vector, torch.zeros_like(mask_vector))
        print('mse loss is : ', mse_loss)
        # print('reg loss is : ', reg_loss)

        loss = mse_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(mask_vector)
    mask_vector = mask_vector.detach().cpu().numpy()


    recon_loss_per_column = np.mean(list_of_recon_loss, axis=0)
    recon_loss_per_column_test = np.mean(list_of_recon_loss_test, axis=0)


    # calculate columnwise kl divergence
    kl_divergences = calculate_columnwise_kl_divergence(dataset.train_x, dataset.test_x, bins=200)
    print('kl divergence is :')
    print(kl_divergences)


    plt.subplot(3, 1, 1)
    plt.bar(range(len(recon_loss_per_column)), kl_divergences)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('KL div per column')
    plt.xticks(range(len(recon_loss_per_column)))



    plt.subplot(3, 1, 2)
    plt.bar(range(len(recon_loss_per_column_test)), np.abs(recon_loss_per_column_test - recon_loss_per_column))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Recon Loss change per column')
    plt.xticks(range(len(recon_loss_per_column_test)))

    mask_vector = (mask_vector - np.min(mask_vector)) / (np.max(mask_vector) - np.min(mask_vector))

    plt.subplot(3, 1, 3)
    bars = plt.bar(range(len(recon_loss_per_column_test)), recon_loss_per_column_test - recon_loss_per_column)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('recon loss per column')
    plt.xticks(range(len(recon_loss_per_column_test)))

    for bar, val in zip(bars, recon_loss_per_column_test - recon_loss_per_column):
        if val >= 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')

    plt.show()
    # plt.subplot(3, 1, 3)
    # plt.bar(range(len(recon_loss_per_column)), recon_loss_per_column)
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('recon loss per column')
    # plt.xticks(range(len(recon_loss_per_column)))
    # plt.show()

    # print the overlap percentage between most important indices of shapely values vs. mask vecotr
    # shap_indices = np.argsort(shap_importance)[::-1][:int(len(shap_importance) * 0.5)]
    # mask_indices = np.argsort(mask_vector)[::-1][:int(len(shap_importance)*0.5)]
    # print(shap_indices)
    # print(mask_indices)
    # print(len(set(shap_indices).intersection(set(mask_indices))) / len(set(shap_indices).union(set(mask_indices))))

    import scipy
    # print(scipy.stats.entropy(mask_vector, shap_importance, base=None, axis=0))


def linear_layer_exp(args, model, dataset):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model = model.to(args.device)
    model.train()

    linear_layer = nn.Sequential(
        nn.Linear(128, 2),
        nn.Softmax(dim=-1)
    )
    linear_layer.to(args.device)
    linear_layer.apply(init_weights)

    params = list(model.parameters()) + list(linear_layer.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)


    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(args.device)
    from utils.utils import get_shap_values
    shap_importance, shap_explainer = get_shap_values(dataset, model,
                                                      torch.tensor(dataset.train_x).float().to(args.device), args)
    shap_importance = np.mean(np.abs(shap_importance), axis=0)

    upper_half = torch.tensor(np.argsort(shap_importance)[len(shap_importance)//4:]).long().to(args.device)
    lower_half = torch.tensor(np.argsort(shap_importance)[:len(shap_importance)//4]).long().to(args.device)

    zeros = torch.zeros_like(torch.tensor(shap_importance)).float().to(args.device)
    upper_mask = zeros
    upper_mask[upper_half] = 1

    lower_mask = zeros
    lower_mask[lower_half] = 1

    for epoch in range(5):
        x = test_x
        y = test_y
        emd_imputed_x = Dataset.get_imputed_data(x, dataset.train_x, data_type="numerical", imputation_method="zero")
        from utils.stunt_prototype import StuntPrototype
        # loss = StuntPrototype.multi_prototype_loss(model, x, emd_imputed_x, shap_importance, num_bins=3)

        teacher_out = F.softmax(model(x), dim=-1)
        feat_x = model.get_feature_shallow(x)
        student_out = linear_layer(feat_x)

        # with_overlap_mask =

        loss_ce = F.cross_entropy(student_out, teacher_out).mean()

        with torch.no_grad():
            student_a = linear_layer(model.get_feature_shallow(x * upper_mask))
            student_b = linear_layer(model.get_feature_shallow(x * lower_mask))

            loss_student_a = F.cross_entropy(student_a, teacher_out).mean()
            loss_student_b = F.kl_div((student_a + 1e-6).log(), student_b, reduction='batchmean')

            print('losses are: {}, {}, {}'.format(loss_ce.item(), loss_student_a.item(), loss_student_b.item()))

        loss = loss_ce
        # loss = softmax_entropy(estim_y).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy in test set
        feat_x = model.get_feature_shallow(x)
        estim_y = linear_layer(feat_x)

        estim = torch.argmax(estim_y, dim=-1)
        acc = (estim == torch.argmax(y, dim=-1)).float().mean()

        print('acc at epoch {} is {}'.format(epoch, acc.item()))
        print('loss at epoch {} is {}'.format(epoch, loss.item()))
        print('')
        # pooled_avg_plot(args, model, dataset)

    # params = list(model.parameters()) + list(linear_layer.parameters())
    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=1e-2)

    for epoch in range(5):
        x = test_x
        y = test_y
        emd_imputed_x = Dataset.get_imputed_data(x, dataset.train_x, data_type="numerical", imputation_method="zero")
        from utils.stunt_prototype import StuntPrototype
        # loss = StuntPrototype.multi_prototype_loss(model, x, emd_imputed_x, shap_importance, num_bins=3)

        teacher_out = F.softmax(model(x), dim=-1)
        student_a = linear_layer(model.get_feature_shallow(x * upper_mask))
        student_b = linear_layer(model.get_feature_shallow(x * lower_mask))

        loss_student_a = F.cross_entropy(student_a, teacher_out).mean()
        # loss_student_b = F.kl_div((student_a + 1e-6).log(), student_b, reduction='batchmean')

        loss = loss_student_a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # accuracy in test set
            feat_x = model.get_feature_shallow(x)
            estim_y = linear_layer(feat_x)

            estim = torch.argmax(estim_y, dim=-1)
            acc = (estim == torch.argmax(y, dim=-1)).float().mean()

            print('acc at epoch {} is {}'.format(epoch, acc.item()))
            print('loss at epoch {} is {}'.format(epoch, loss.item()))
            print('')
            # pooled_avg_plot(args, model, dataset)



def prediction_based_on_mask(args, model, dataset):
    model = model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    dropout = nn.Dropout(p=0.75)


    model.train()
    model.requires_grad_(False)

    from utils.utils import get_shap_values
    shap_importance, shap_explainer = get_shap_values(dataset, model,
                                                      torch.tensor(dataset.train_x).float().to(args.device), args)
    shap_importance = np.mean(np.abs(shap_importance), axis=0)


    # generate class prototypes
    class_prototypes_train = []
    features_train = model.get_feature_shallow(train_x)
    for label in range(dataset.train_y.shape[1]):
        idx = train_y[:, label] == 1
        class_prototypes_train.append(torch.mean(features_train[idx], 0))
    print(class_prototypes_train)

    class_prototypes_test = []
    features_test = model.get_feature_shallow(test_x)
    for label in range(dataset.test_y.shape[1]):
        idx = test_y[:, label] == 1
        class_prototypes_test.append(torch.mean(features_test[idx], 0))
    print(class_prototypes_test)

    class_prototypes_train_np = [class_prototypes_train[0].detach().cpu().numpy(), class_prototypes_train[1].detach().cpu().numpy()]
    class_prototypes_test_np = [class_prototypes_test[0].detach().cpu().numpy(), class_prototypes_test[1].detach().cpu().numpy()]
    tsne = TSNE(n_components=2, random_state=0, perplexity=2)
    X_2d = tsne.fit_transform(np.array([class_prototypes_train_np[0], class_prototypes_train_np[1],class_prototypes_test_np[0],class_prototypes_test_np[1]]))

    # visualize tsne
    plt.figure(figsize=(6, 5))
    colors = ['r', 'b', 'g', 'y']
    target_ids = range(4)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100)
    plt.legend(['train_0', 'train_1', 'test_0', 'test_1'])
    plt.show()

    label_list = []
    for idx in range(len(dataset.test_x)):
        mask_vector_zero = torch.ones_like(train_x[0]).float().to(args.device)
        mask_vector_zero = mask_vector_zero.requires_grad_(True)
        optimizer = torch.optim.Adam([mask_vector_zero], lr=1e-1)

        for epoch in range(10):
            single_feature = model.get_feature_shallow(mask_vector_zero * test_x[idx])
            mse_loss_fn = nn.MSELoss()
            loss = mse_loss_fn(single_feature, class_prototypes_train[0]) * 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f'loss at epoch {epoch} is {loss.item()}')

        loss_zero = loss.item()

        mask_vector_one = torch.ones_like(train_x[0]).float().to(args.device)
        mask_vector_one = mask_vector_one.requires_grad_(True)
        optimizer = torch.optim.Adam([mask_vector_one], lr=1e-1)
        for epoch in range(10):
            single_feature = model.get_feature_shallow(mask_vector_one * test_x[idx])
            mse_loss_fn = nn.MSELoss()
            loss = mse_loss_fn(single_feature, class_prototypes_train[1]) * 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f'loss at epoch {epoch} is {loss.item()}')

        loss_one = loss.item()

        norm_mask_vector_zero = (mask_vector_zero - torch.min(mask_vector_zero)) / (torch.max(mask_vector_zero) - torch.min(mask_vector_zero))
        norm_mask_vector_one = (mask_vector_one - torch.min(mask_vector_one)) / (torch.max(mask_vector_one) - torch.min(mask_vector_one))

        norm_mask_vector_one = norm_mask_vector_one.cpu().detach().numpy()
        norm_mask_vector_zero = norm_mask_vector_zero.cpu().detach().numpy()

        shap_indices = np.argsort(shap_importance)[::-1][:int(len(shap_importance) * 0.5)]
        mask_indices_zero = np.argsort(norm_mask_vector_zero)[::-1][:int(len(shap_importance) * 0.1)]
        mask_indices_one = np.argsort(norm_mask_vector_one)[::-1][:int(len(shap_importance) * 0.1)]

        intersection_zero = len(set(shap_indices).intersection(set(mask_indices_zero))) / len(set(shap_indices).union(set(mask_indices_zero)))
        intersection_one = len(set(shap_indices).intersection(set(mask_indices_one))) / len(set(shap_indices).union(set(mask_indices_one)))

        print(sorted(mask_indices_one))
        print(sorted(mask_indices_zero))
        print('')

        if intersection_zero > intersection_one:
            label_list.append(0)
        else:
            label_list.append(1)

        # if loss_zero > loss_one:
        #     label_list.append(1)
        # else:
        #     label_list.append(0)

    # print accuracy between dataset.train_y and label_list
    print('accuracy')
    print((np.array(label_list) == np.argmax(dataset.test_y, axis=-1)).mean())

def interpolation(args, model, dataset):

    model = model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    dropout = nn.Dropout(p=0.75)


    model.train()
    model.requires_grad_(False)

    from utils.utils import get_shap_values
    shap_importance, shap_explainer = get_shap_values(dataset, model,
                                                      torch.tensor(dataset.train_x).float().to(args.device), args)
    shap_importance = np.mean(np.abs(shap_importance), axis=0)
    all = torch.tensor(list(np.argsort(shap_importance)[::-1][:int(len(shap_importance) * 1)])).long()
    upper25 = torch.tensor(list(np.argsort(shap_importance)[::-1][:int(len(shap_importance) * 0.3)])).long()
    lower25 = torch.tensor(list(np.argsort(shap_importance)[:int(len(shap_importance) * 0.3)])).long()
    emd_imputed_test_x = Dataset.get_imputed_data(test_x, dataset.train_x, data_type="numerical", imputation_method="emd")
    emd_imputed_train_x = Dataset.get_imputed_data(train_x, dataset.train_x, data_type="numerical", imputation_method="emd")

    for idx in [all, upper25, lower25]:
        # dropout = nn.Dropout(p=mask_ratio)
        mask = torch.zeros_like(train_x[0]).float().to(args.device)
        mask[idx] = 1

        masked_feature = model.get_feature(mask * train_x)

        tsne = TSNE(n_components=2, random_state=0)

        masked_feature = masked_feature.cpu().detach().numpy()
        # l2 normalize masked features
        masked_feature = masked_feature / np.linalg.norm(masked_feature, axis=-1, keepdims=True)

        masked_feature = tsne.fit_transform(masked_feature)

        # # for each label, scatter:
        # for label in range(2):
        #     plt.scatter(masked_feature[:, 0], masked_feature[:, 1], c=np.argmax(train_y.cpu().detach().numpy(), axis=-1))
        #     plt.show()

        plt.scatter(masked_feature[:, 0], masked_feature[:, 1], c=np.argmax(train_y.cpu().detach().numpy(), axis=-1))
        plt.legend()
        plt.show()
    # recon_x = model.get_recon_out(test_x)
    orig_y = model(test_x)
    orig_y = orig_y.cpu().detach().numpy()
    orig_y = np.argmax(orig_y, axis=-1)

    print('accuracy')
    print((orig_y == np.argmax(dataset.test_y, axis=-1)).mean())

    for idx in upper25:
        mask = torch.zeros_like(train_x[0]).float().to(args.device)
        mask[idx] = 1

        upper_mask = torch.zeros_like(train_x[0]).float().to(args.device)
        upper_mask[upper25] = 1

        recon_x = model.get_recon_out(test_x * (1 - upper_mask))
        sourced_x = recon_x * upper_mask + test_x * (1 - upper_mask)

        # plot historgrams of train_x[idx] test_x[idx]
        plt.hist(train_x[:, idx].cpu().detach().numpy(), bins=100, alpha=0.5, label='train')
        plt.title(f'feature {idx}, train')
        plt.show()
        plt.hist(test_x[:, idx].cpu().detach().numpy(), bins=100, alpha=0.5, label='test')
        plt.title(f'feature {idx}, test')
        plt.show()
        plt.hist(recon_x[:, idx].cpu().detach().numpy(), bins=100, alpha=0.5, label='test')
        plt.title(f'feature {idx}, recon')
        plt.show()

    sourced_y = model(sourced_x)
    sourced_y = sourced_y.cpu().detach().numpy()

    sourced_y = np.argmax(sourced_y, axis=-1)
    # print accuracy between dataset.train_y and label_list
    print((sourced_y == np.argmax(dataset.test_y, axis=-1)).mean())

    sourced_y = model(model.get_recon_out(test_x))
    sourced_y = sourced_y.cpu().detach().numpy()

    sourced_y = np.argmax(sourced_y, axis=-1)
    # print accuracy between dataset.train_y and label_list
    print((sourced_y == np.argmax(dataset.test_y, axis=-1)).mean())

def cont_cat_features(args, model, dataset):
    test_cont_rescaled = dataset.test_cont_x_scaled
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    # generate cont mask, cat mask
    cont_mask = torch.zeros_like(train_x[0]).float().to(args.device)
    cont_mask[:test_cont_rescaled.shape[1]] = 1

    cat_mask = 1 - cont_mask


    feature_cont = model.get_feature(train_x * cont_mask)
    feature_cat = model.get_feature(train_x * cat_mask)

    feature_cont = feature_cont.cpu().detach().numpy()
    feature_cat = feature_cat.cpu().detach().numpy()

    # tsne
    tsne = TSNE(n_components=2, random_state=0)
    feature_cont_tsne = tsne.fit_transform(feature_cont)

    tsne = TSNE(n_components=2, random_state=0)
    feature_cat_tsne = tsne.fit_transform(feature_cat)

    tsne = TSNE(n_components=2, random_state=0)
    feature_contcat = tsne.fit_transform(np.concatenate([feature_cont, feature_cat], axis=0))
    feature_contcat_cmap = np.concatenate([np.zeros_like(feature_cont[:, 0]), np.ones_like(feature_cat[:, 0])], axis=0)
    feature_contcat_labelcmap = np.concatenate([np.argmax(dataset.train_y, axis=-1), np.argmax(dataset.train_y, axis=-1)], axis=0)
    # get recon_x
    plt.figure(figsize=(10, 30))
    plt.subplot(4, 1, 1)
    plt.scatter(feature_cont_tsne[:, 0], feature_cont_tsne[:, 1], c=np.argmax(train_y.cpu().detach().numpy(), axis=-1), cmap='rainbow')
    plt.title('Continuous feature')

    plt.subplot(4, 1, 2)
    plt.scatter(feature_cat_tsne[:, 0], feature_cat_tsne[:, 1], c=np.argmax(train_y.cpu().detach().numpy(), axis=-1), cmap='rainbow')
    plt.title('Categorical features')

    plt.subplot(4, 1, 3)
    plt.scatter(feature_contcat[:, 0], feature_contcat[:, 1], c=feature_contcat_cmap, cmap='rainbow')
    plt.title('Together')

    plt.subplot(4, 1, 4)
    plt.scatter(feature_contcat[:, 0], feature_contcat[:, 1], c=feature_contcat_labelcmap, cmap='rainbow')
    plt.title('Together in labels')
    plt.show()

def conv_features(args, dataset, model):
    conv_w_pooling = nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.AvgPool1d(4, 1),
    )
    conv_w_pooling.to(args.device)
    conv_w_pooling.requires_grad_(True)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    optimizer = torch.optim.AdamW(conv_w_pooling.parameters(), lr=0.001)

    model = model.to(args.device)
    model.requires_grad_(False)
    xgb_importance = get_xgb_feature_importance(args, dataset)
    xgb_importance = torch.argsort(torch.tensor(xgb_importance), descending=True)

    top_mask = torch.zeros_like(train_x[0]).float().to(args.device)
    lower_mask = torch.zeros_like(train_x[0]).float().to(args.device)

    top_mask[xgb_importance[:int(train_x.shape[1]/5)]] = 1
    lower_mask[xgb_importance[int(train_x.shape[1]/5):]] = 1

    for epoch in range(100):
        initial_feature = model.get_feature(test_x)
        initial_feature_shallow = model.get_feature_shallow(test_x)

        upper_train_x = test_x * top_mask
        input_conv_upper = conv_w_pooling(test_x.unsqueeze(1))
        input_conv_upper = input_conv_upper.squeeze()
        upper_feature = model.get_feature(input_conv_upper)
        upper_feature_shallow = model.get_feature_shallow(input_conv_upper)

        lower_train_x = test_x * lower_mask
        input_conv_lower = conv_w_pooling(test_x.unsqueeze(1))
        input_conv_lower = input_conv_lower.squeeze()
        lower_feature = model.get_feature(input_conv_lower)
        lower_feature_shallow = model.get_feature_shallow(input_conv_lower)

        optimizer.zero_grad()
        # mse loss between initial feature and upper feature, while increased distance between initial and lower feature
        loss =  F.mse_loss(initial_feature, upper_feature) + F.mse_loss(initial_feature, lower_feature) + F.mse_loss(initial_feature_shallow, upper_feature_shallow)
        print(f'loss is : {loss}')
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        feature_torch = model.get_feature(conv_w_pooling(test_x.unsqueeze(1)).squeeze())
        feature = feature_torch.cpu().detach().numpy()
        # tsne
        tsne = TSNE(n_components=2, random_state=0)
        feature_tsne = tsne.fit_transform(feature)

        plt.figure(figsize=(10, 10))
        plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=np.argmax(test_y.cpu().detach().numpy(), axis=-1), cmap='rainbow')
        plt.show()

        estim_y = model(conv_w_pooling(test_x.unsqueeze(1)).squeeze())
        # accuracy
        np_estim_y = torch.argmax(estim_y, dim=-1).detach().cpu().numpy()
        print(np_estim_y[0])
        print(np.unique(np_estim_y, return_counts=True))
        print(
            f"acc {(torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item()}")

def plot_permutation_result(args, dataset, model):

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    model = model.to(args.device)
    model.requires_grad_(False)
    model.eval()
    model.requires_grad_(True)

    # init_model = init_model.to(args.device)
    # init_model.eval()
    # init_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(100):
        unique_permutations = {tuple(perm): i for i, perm in enumerate(itertools.permutations(range(args.num_chunks)))}
        from utils.permutation import Permutation_Generation
        permute_train_x, permute_train_y = Permutation_Generation.permutation_task_generation(train_x,
                                                                                              unique_permutations,
                                                                                              args.num_chunks,
                                                                                              args.num_chunks_per_sample)
        permute_test_x, permute_test_y = Permutation_Generation.permutation_task_generation(test_x, unique_permutations,
                                                                                            args.num_chunks,
                                                                                            args.num_chunks_per_sample)

        permute_train_x, permute_train_y = permute_train_x.to(args.device).float(), permute_train_y.to(args.device)
        permute_test_x, permute_test_y = permute_test_x.to(args.device).float(), permute_test_y.to(args.device)


        optimizer.zero_grad()
        (F.cross_entropy(model.get_ordering(permute_test_x), permute_test_y).mean()).backward()
        optimizer.step()
        with torch.no_grad():
            # accuracy of permutation tasks on train and test
            train_acc = (torch.argmax(model.get_ordering(permute_train_x), dim=-1) == permute_train_y).float().mean().item()
            test_acc = (torch.argmax(model.get_ordering(permute_test_x), dim=-1) == permute_test_y).float().mean().item()

            print(f"train acc is {train_acc}, test acc is {test_acc}")
            permutation_invariant(args, dataset, model)

def permutation_invariant(args, dataset, model):
    model = model.to(args.device)
    with torch.no_grad():
        train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(
            dataset.train_y).float().to(
            args.device)
        test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
            args.device)

        train_x_permuted = permute_columns(train_x)
        test_x_permuted = permute_columns(test_x)

        # print accuracy without permutation
        train_acc = (torch.argmax(model(train_x), dim=-1) == torch.argmax(train_y, dim=-1)).float().mean().item()
        test_acc = (torch.argmax(model(test_x), dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item()
        print(f"wo permutation train acc is {train_acc}, test acc is {test_acc}")

        # print accuracy upon permutation
        train_acc = (torch.argmax(model(train_x_permuted), dim=-1) == torch.argmax(train_y, dim=-1)).float().mean().item()
        test_acc = (torch.argmax(model(test_x_permuted), dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item()
        print(f"w permutation train acc is {train_acc}, test acc is {test_acc}")


def permuation_invariant(args, dataset, model):
    model = model.to(args.device)
    model.train()
    model.requires_grad_(True)

    import copy
    orig_model = copy.deepcopy(model)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    for epoch in range(3):
        optimizer.zero_grad()
        feature_train = orig_model.get_feature(test_x)
        feature_permuted = model.get_feature(permute_columns(test_x))
        # feature_permuted_2 = model.get_feature(permute_columns(test_x))

        loss = F.mse_loss(feature_train.detach(), feature_permuted)
        # loss += F.mse_loss(feature_train.detach(), feature_permuted_2)
        loss.backward()
        optimizer.step()

        print(f'loss is : {loss}')

        # update orig_model parameters in an ema fashion
        for param, orig_param in zip(model.parameters(), orig_model.parameters()):
            orig_param.data = orig_param.data * 0.9 + param.data * 0.1
        orig_model.requires_grad_(False)

        with torch.no_grad():
            # accuracy:
            estim_y = model(test_x)
            np_estim_y = torch.argmax(estim_y, dim=-1).detach().cpu().numpy()
            print(np_estim_y[0])
            print(np.unique(np_estim_y, return_counts=True))
            print('acc : ', (torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())
    plot_permutation_result(args, dataset, model)


def plot_gumbel_softmax(args, dataset, model):

    model = model.to(args.device)
    model.eval()
    model.requires_grad_(True)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(args.device)

    single_training_sample = train_x[0]

    class GumbelBinaryMask(nn.Module):
        def __init__(self, feature_count, tau=1.0):
            super(GumbelBinaryMask, self).__init__()
            self.feature_count = feature_count
            self.tau = tau

            # Initialize logits such that each column sums to 1 (since it's a 2-class softmax)
            logits = torch.randn(feature_count, 2)
            logits -= logits.logsumexp(-1, keepdim=True)
            self.logits = nn.Parameter(logits)

        def forward(self):
            soft_mask = F.gumbel_softmax(self.logits, tau=self.tau, hard=False)[:, 0]

            # To mask out the bottom 75% of the features based on the soft mask
            threshold = torch.quantile(soft_mask, 0.25)
            hard_mask = (soft_mask <= threshold).float()

            return hard_mask

    mask_layer = GumbelBinaryMask(len(single_training_sample), tau=1)
    mask_layer = mask_layer.to(args.device)
    optimizer = torch.optim.AdamW(mask_layer.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        masked_sample = train_x * mask_layer()
        loss = - F.l1_loss(model.get_recon_out(masked_sample), train_x)
        loss.backward()
        optimizer.step()
        print('loss at epoch {} is {}'.format(epoch, loss.item()))
        with torch.no_grad():
            print('masked_sample is {}'.format(mask_layer().detach().cpu().numpy()))

            # plot bar graph of masked sample
            plt.bar(np.arange(len(mask_layer().detach().cpu().numpy())), mask_layer().detach().cpu().numpy())
            plt.show()

def use_avg_feature(args, dataset, model):
    model = model.to(args.device)
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    model.eval()
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    total_epochs = 25
    for epoch in range(total_epochs):
        mask = torch.rand(test_x.shape) < 1 - 0.9 * (epoch / total_epochs)
        mask = mask.to(args.device)
        masked_x = test_x * mask.float().to(args.device)
        masked_output_feature = model.get_recon_out(masked_x)

        with torch.no_grad():
            crit = nn.MSELoss(reduction='none')
            mse_loss_list = torch.sum(crit(masked_output_feature * (~mask), test_x * (~mask)), dim=-1)
            mse_idx = torch.argsort(mse_loss_list, descending=True)[:len(test_x) // 10]

        optimizer.zero_grad()
        F.mse_loss(masked_output_feature[mse_idx] * (~mask[mse_idx]), test_x[mse_idx] * (~mask[mse_idx])).backward()
        optimizer.step()

        with torch.no_grad():
            vanilla_output = model(test_x)

            # get average feature
            feature_list = []
            for mask_ratio in [0.25, 0.5, 0.75]:
                mask = torch.rand(test_x.shape) < mask_ratio
                masked_x = test_x * mask.float().to(args.device)
                masked_output_feature = model.get_feature(masked_x)
                feature_list.append(masked_output_feature)
            avg_feature = torch.mean(torch.stack(feature_list), dim=0)

            averaged_output = model.cls_head(avg_feature)

            # vanilla accuracy
            print(f'epoch : {epoch} vanilla accuracy : ',
                  (torch.argmax(vanilla_output, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())
            print(f'epoch : {epoch} averaged accuracy : ',
                  (torch.argmax(averaged_output, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())

    print('\n\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        entropy = softmax_entropy(model(test_x) / 0.5, dim=-1).mean()
        print(f'epoch : {epoch} entropy : ', entropy.item())
        entropy.backward()
        optimizer.step()

        with torch.no_grad():
            vanilla_output = model(test_x)

            # get average feature
            feature_list = []
            for mask_ratio in [0.25, 0.5, 0.75]:
                mask = torch.rand(test_x.shape) < mask_ratio
                masked_x = test_x * mask.float().to(args.device)
                masked_output_feature = model.get_feature(masked_x)
                feature_list.append(masked_output_feature)
            avg_feature = torch.mean(torch.stack(feature_list), dim=0)

            averaged_output = model.cls_head(avg_feature)

            # vanilla accuracy
            print(f'epoch : {epoch} vanilla accuracy : ',
                  (torch.argmax(vanilla_output, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())
            print(f'epoch : {epoch} averaged accuracy : ',
                  (torch.argmax(averaged_output, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())

def smotish(args, dataset, model):
    def contrastive_loss(features, group_indices):

        # normalize features
        features = F.normalize(features, dim=1)

        # Step 1: Group Inputs
        groups = [[] for _ in range(np.unique(group_indices).shape[0])]
        for idx in range(features.shape[0]):
            group = group_indices[idx]
            groups[group].append(idx)

        # Step 5: Within-Group Loss
        within_group_loss = 0
        for group in groups:
            group_features = features[group]
            centroid = group_features.mean(dim=0, keepdim=True)
            distances = ((group_features - centroid) ** 2).sum(dim=1)
            within_group_loss += distances.mean()

        # Step 6: Between-Group Loss
        centroids = [features[group].mean(dim=0) for group in groups]
        centroids = torch.stack(centroids)
        centroid_distances = ((centroids[:, None] - centroids[None, :]) ** 2).sum(dim=2)
        between_group_loss = centroid_distances.mean()

        # Step 7: Combine Losses
        combined_loss = - between_group_loss

        return combined_loss

    model.train()
    model.requires_grad_(True)
    model.to(args.device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(20, test_x.shape[0]))
    kmeans.fit(test_x.detach().cpu().numpy())
    group_labels = kmeans.labels_

    # use smote to upsample
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    upsampled_x, upsampled_group_labels = smote.fit_resample(test_x.detach().cpu().numpy(), group_labels)
    upsampled_x = torch.tensor(test_x).float().to(args.device)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = contrastive_loss(model.get_feature(upsampled_x), upsampled_group_labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            # plot acc
            print(f'epoch : {epoch} loss : ', loss.item())
            print(f'epoch : {epoch} accuracy : ', (torch.argmax(model(test_x), dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item())

            if epoch % 10 == 0:
                # plot the features in tsne
                from sklearn.manifold import TSNE
                plt.figure(figsize=(30, 20))

                plt.subplot(3, 2, 1)
                tsne = TSNE(n_components=2)
                feature = model.get_feature(test_x)
                feature = F.normalize(feature, dim=1).detach().cpu().numpy()
                tsne_features = tsne.fit_transform(feature)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=torch.argmax(test_y, dim=-1).detach().cpu().numpy())
                plt.title('deep features at epoch {}'.format(epoch))
                # plt.show()

                plt.subplot(3, 2, 2)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=group_labels)
                plt.title('deep w group labels at epoch {}'.format(epoch))
                # plt.show()


                plt.subplot(3, 2, 3)
                tsne = TSNE(n_components=2)
                feature = model.get_feature_shallow(test_x)
                feature = F.normalize(feature, dim=1).detach().cpu().numpy()
                tsne_features = tsne.fit_transform(feature)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                            c=torch.argmax(test_y, dim=-1).detach().cpu().numpy())
                plt.title('shallow features at epoch {}'.format(epoch))
                # plt.show()

                plt.subplot(3, 2, 4)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=group_labels)
                plt.title('shallow w group labels at epoch {}'.format(epoch))
                # plt.show()

                plt.subplot(3, 2, 5)
                tsne = TSNE(n_components=2)
                feature = test_x
                feature = F.normalize(feature, dim=1).detach().cpu().numpy()
                tsne_features = tsne.fit_transform(feature)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                            c=torch.argmax(test_y, dim=-1).detach().cpu().numpy())
                plt.title('input features at epoch {}'.format(epoch))
                # plt.show()

                plt.subplot(3, 2, 6)
                plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=group_labels)
                plt.title('input w group labels at epoch {}'.format(epoch))
                plt.show()

def plot_shap_per_column(args, dataset, source_model, test_model):
    from utils.utils import get_shap_values
    shap_importance_train, _ = get_shap_values(dataset, source_model,
                                                      torch.tensor(dataset.train_x).float().to(args.device), args)

    shap_importance_test, _ = get_shap_values(dataset, test_model, torch.tensor(dataset.test_x).float().to(args.device), args)


    n_class = dataset.test_y.shape[1]
    plt.figure(figsize=(10, 5 * n_class))
    for idx, shap_per_class in enumerate(shap_importance_train):
        plt.subplot(n_class, 1, 1 + idx)
        plt.bar(range(len(shap_per_class)), shap_per_class)
        plt.xlabel('Index')
        plt.ylabel('Shap values')
        plt.title(f'Shap values - TRAIN of class {idx}')
        plt.xticks(range(len(shap_per_class)))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5 * n_class))
    for idx, shap_per_class in enumerate(shap_importance_test):
        plt.subplot(n_class, 1, 1 + idx)
        plt.bar(range(len(shap_per_class)), shap_per_class)
        plt.xlabel('Index')
        plt.ylabel('Shap values')
        plt.title(f'Shap values - TEST of class {idx}')
        plt.xticks(range(len(shap_per_class)))
    plt.tight_layout()
    plt.show()

def give_adaptive_recon_loss(args, dataset, source_model, test_model):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    source_model.train()
    source_model.requires_grad_(True)
    optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.0001)

    with torch.no_grad():
        dropout = nn.Dropout(p=0.75)
        recon_out = source_model.get_recon_out(dropout(test_x))
        recon_loss = F.mse_loss(recon_out, test_x, reduction='none').mean(0)
        sorted_column_via_recon_loss = torch.argsort(recon_loss, descending=True)

        mask = torch.ones_like(recon_loss).to(args.device)
        mask[sorted_column_via_recon_loss[:int(len(recon_loss) * 0.5)]] = 0

    buckets = 3
    idx_tensor_list = torch.tensor([idx for idx in range(len(recon_loss))])
    split_idx_tensor_list = torch.split(idx_tensor_list, int(len(recon_loss) * 1/buckets))

    masks = []
    for split_idx_tensor in split_idx_tensor_list:
        mask = torch.zeros_like(recon_loss).to(args.device)
        mask[split_idx_tensor] = 1
        masks.append(mask)

    from utils.sparsemax import Sparsemax
    sparsemax = Sparsemax(dim=-1)
    reciprocal = torch.tensor(torch.reciprocal(recon_loss + 1e-6)).to(args.device)
    alpha = sparsemax(torch.tensor(reciprocal) / torch.sum(reciprocal))
    alpha = 0.9 * alpha / alpha.max()

    for epoch in range(300):
        optimizer.zero_grad()

        low_recon_loss_columns = sorted_column_via_recon_loss[int(len(recon_loss) * 0.1):]  # Set an appropriate threshold

        # Step 2: Finding the Nearest Neighbor
        from sklearn.neighbors import NearestNeighbors

        nearest_neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(dataset.test_x)
        distances, indices = nearest_neighbors.kneighbors(dataset.test_x)

        # Step 3: Mixup Augmentation
        # alpha = 0.1  # Set an appropriate mixup coefficient
        augmented_batch = []
        for i, test_instance in enumerate(test_x):
            nearest_neighbor = test_x[
                indices[i, 1]]  # Getting the 2nd nearest neighbor because the nearest would be the instance itself

            nearest_neighbor = torch.tensor(nearest_neighbor).to(args.device)
            # Only considering columns with low reconstruction loss for mixup
            # mixup_instance = test_instance.clone()
            mixup_instance = test_instance * alpha + nearest_neighbor * (1 - alpha)

            augmented_batch.append(mixup_instance)

        augmented_batch = torch.stack(augmented_batch).float().to(args.device)


        out = source_model(test_x)
        out_aug = source_model(augmented_batch)

        loss = F.cross_entropy(out, torch.softmax(out / 0.8, dim=1).detach()).mean() + F.cross_entropy(out_aug, torch.softmax(out_aug / 0.8, dim=1).detach()).mean()
        loss.backward()


        # loss = -F.mse_loss(F.normalize(feat, dim=1), F.normalize(feat_aug, dim=1)).mean()
        # loss.backward()


        # test_out_loss = softmax_entropy(source_model(test_x) / 0.5).mean()
        # aug_test_out = softmax_entropy(source_model(augmented_batch) / 0.5).mean()
        #
        # loss = (test_out_loss + aug_test_out)
        # loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            # print('epoch {}, loss {}'.format(epoch, loss.item()))
            # print acc
            source_model.eval()
            with torch.no_grad():
                test_out = torch.softmax(source_model(test_x), dim=-1)
                aug_test_out = torch.softmax(source_model(augmented_batch), dim=-1)

                print(f'prediction softmax of test: {test_out[0]}')
                print(f'prediction softmax of augmented test: {aug_test_out[0]}')

                test_acc = (torch.argmax(test_out, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean()
                aug_test_acc = (torch.argmax(aug_test_out, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean()
                print(f'epoch: {epoch} test acc {test_acc.item()}')
                print(f'epoch: {epoch} aug test acc {aug_test_acc.item()}')

                print(f'count of elems are: {np.unique(torch.argmax(test_out, dim=-1).detach().cpu().numpy(), return_counts=True)}')
                print(f'count of augelems are: {np.unique(torch.argmax(aug_test_out, dim=-1).detach().cpu().numpy(), return_counts=True)}')
                print('\n')
            source_model.train()

def subtract_emd(args, model, dataset):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    emd_imputed_train_x = Dataset.get_imputed_data(train_x, dataset.train_x, data_type="numerical",
                                                   imputation_method="mean")
    emd_imputed_train_x = torch.tensor(emd_imputed_train_x).float().to(args.device)
    emd_imputed_test_x = Dataset.get_imputed_data(test_x, dataset.test_x, data_type="numerical",
                                                  imputation_method="mean")
    emd_imputed_test_x = torch.tensor(emd_imputed_test_x).float().to(args.device)

    model.eval()
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # generate class prototypes
    class_prototypes = []
    features_train = model.get_feature(train_x)
    for label in range(dataset.train_y.shape[1]):
        idx = train_y[:, label] == 1
        class_prototypes.append(torch.mean(features_train[idx], 0))
    print(class_prototypes)
    class_prototypes = torch.stack(class_prototypes).to(args.device)


    for epoch in range(300):

        get_features_naive = model.get_feature(test_x)
        # get_features_emd_train = model.get_feature(emd_imputed_train_x)[0].repeat(get_features_naive.shape[0], 1)
        # get_features_emd_test = model.get_feature(emd_imputed_test_x)[0].repeat(get_features_naive.shape[0], 1)
        #
        from utils.uniformity_loss import uniformity_loss
        loss = uniformity_loss(get_features_naive)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print acc
        with torch.no_grad():
            train_features = model.get_feature(train_x)
            test_features = model.get_feature(test_x)


            def recalibration(train_features, test_features, threshold):
                train_mean = torch.mean(train_features, dim=0)
                train_features_centered = train_features - train_mean
                train_cov_matrix = torch.matmul(train_features_centered.T, train_features_centered) / (
                            train_features.size(0) - 1)
                train_cov_matrix += torch.eye(train_cov_matrix.size(0)).to(args.device) * 1e-6
                train_cov_matrix_inv = torch.pinverse(train_cov_matrix)

                # Step 2: Compute the mean of the test data features
                test_mean = torch.mean(test_features, dim=0)

                # Initialize a tensor to store the recalibrated test data features
                recalibrated_test_features = torch.zeros_like(test_features)

                list_dist = []
                for i, test_feature in enumerate(test_features):
                    # Calculate the Mahalanobis distance
                    diff = test_feature - train_mean
                    mahalanobis_distance = torch.sqrt(torch.matmul(torch.matmul(diff.T, train_cov_matrix_inv), diff))
                    list_dist.append(mahalanobis_distance.item())
                    # Step 3: Apply recalibration based on the Mahalanobis distance
                    if mahalanobis_distance > threshold:
                        recalibrated_test_features[i] = test_feature - train_mean + test_mean
                    else:
                        recalibrated_test_features[i] = test_feature

                print(f'mean of mahalanobis distance is {np.mean(list_dist)}')
                print(f'median of mahalanobis distance is {np.median(list_dist)}')
                return recalibrated_test_features

            from scipy.stats import chi2

            # Calculate degrees of freedom
            degrees_of_freedom = train_features.shape[1]  # Number of features

            # Calculate the threshold for Mahalanobis distance
            # Here, 0.6827 is the cumulative distribution function (CDF) value for a variance of 1 (or one standard deviation)
            threshold = chi2.ppf(0.6827, degrees_of_freedom)

            # Convert the threshold back to Mahalanobis distance (take the square root)
            threshold = torch.sqrt(torch.tensor(threshold))

            recalibrated_test_features = recalibration(train_features, test_features, threshold)
            out = model.cls_head(recalibrated_test_features)

            test_acc = (torch.argmax(out, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean()
            list_of_preds = torch.argmax(out, dim=-1).detach().cpu().numpy()

            print(f'epoch: {epoch} test acc {test_acc.item()}')
            print(f'epoch: {epoch}, preds {np.unique(list_of_preds, return_counts=True)}')
            print('\n')

def plot_distance_over_acc(args, dataset, model):
    def recalibration(train_features, test_features, threshold):
        train_mean = torch.mean(train_features, dim=0)
        train_features_centered = train_features - train_mean
        train_cov_matrix = torch.matmul(train_features_centered.T, train_features_centered) / (
                train_features.size(0) - 1)
        train_cov_matrix += torch.eye(train_cov_matrix.size(0)).to(args.device) * 1e-6
        train_cov_matrix_inv = torch.inverse(train_cov_matrix)

        # Step 2: Compute the mean of the test data features
        test_mean = torch.mean(test_features, dim=0)

        # Initialize a tensor to store the recalibrated test data features
        recalibrated_test_features = torch.zeros_like(test_features)

        list_dist = []
        for i, test_feature in enumerate(test_features):
            # Calculate the Mahalanobis distance
            diff = test_feature - train_mean
            mahalanobis_distance = torch.sqrt(torch.matmul(torch.matmul(diff.T, train_cov_matrix_inv), diff))
            list_dist.append(mahalanobis_distance.item())
            # Step 3: Apply recalibration based on the Mahalanobis distance
            # print(list_dist[-1])
            if mahalanobis_distance > threshold:
                recalibrated_test_features[i] = test_feature - train_mean + test_mean
            else:
                recalibrated_test_features[i] = test_feature

        print(f'mean of mahalanobis distance is {np.mean(list_dist)}')
        print(f'median of mahalanobis distance is {np.median(list_dist)}')
        return list_dist

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    train_features = model.get_feature(train_x)
    test_features = model.get_feature(test_x)
    from scipy.stats import chi2

    # Calculate degrees of freedom
    degrees_of_freedom = train_features.shape[1]  # Number of features

    # Calculate the threshold for Mahalanobis distance
    # Here, 0.6827 is the cumulative distribution function (CDF) value for a variance of 1 (or one standard deviation)
    threshold = chi2.ppf(0.6827, degrees_of_freedom)

    # Convert the threshold back to Mahalanobis distance (take the square root)
    threshold = torch.sqrt(torch.tensor(threshold))

    print('threshold', threshold)
    distance = recalibration(train_features, test_features, threshold)

    out = model(test_x)
    test_acc = (torch.argmax(out, dim=-1) == torch.argmax(test_y, dim=-1)).float()

    def plot_accuracy(mahalanobis_distances, predictions):
        # Step 1: Create bins based on Mahalanobis distances using quantiles
        num_bins = 5
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = np.quantile(mahalanobis_distances, quantiles)

        # Step 2: Assign each sample to a bin based on its Mahalanobis distance
        bin_indices = np.digitize(mahalanobis_distances, bins)

        # Step 3: Calculate accuracy for each bin
        bin_accuracies = []
        for i in range(num_bins):
            bin_samples = np.where(bin_indices == i)[0]
            if len(bin_samples) > 0:
                bin_accuracy = np.mean(predictions[bin_samples])
                bin_accuracies.append(bin_accuracy)
            else:
                bin_accuracies.append(None)

        # Step 4: Plot the accuracy for each bin
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.figure()
        plt.plot(bin_centers, bin_accuracies, marker='o')
        plt.xlabel('Mahalanobis Distance')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Mahalanobis Distance Bins')
        plt.grid(True)
        plt.show()

    plot_accuracy(distance, test_acc.detach().cpu().numpy())

def recon_wise_correction(args, dataset, model):
    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)

    model.requires_grad_(False)
    pred_list = []
    # for idx, test_sample in enumerate(test_x):
    test_feature = model.get_feature(test_x).detach()

    gamma = torch.ones_like(test_feature[0], requires_grad=True).to(args.device)
    beta = torch.zeros_like(test_feature[0], requires_grad=True).to(args.device)

    optimizer = torch.optim.AdamW([gamma, beta], lr=0.01)

    for epoch in range(10):
        out = model.main_head(test_feature * gamma + beta)
        # loss = torch.mean(torch.pow(recon_out_single - test_sample, 2))

        loss = F.cross_entropy(out, torch.argmax(test_y, dim=-1)).mean()

        optimizer.zero_grad()
        loss.backward()
        print(f'epoch {epoch} loss is {loss}')
        optimizer.step()

    print(f'gamma - with gt is {gamma}')
    print(f'beta - with gt is {beta}')
    print('\n')

    # use the gamma and beta to reconstruct the test sample
    out = model.main_head(test_feature * gamma + beta)
    pred_list.append(out)

    pred_list = torch.stack(pred_list)
    test_acc = (torch.argmax(pred_list, dim=-1) == torch.argmax(test_y, dim=-1)).float()
    print(f'test acc - gt is {torch.mean(test_acc)}')

    pred_list = []
    do = nn.Dropout(0.75)
    test_feature = model.get_feature(do(test_x)).detach()


    gamma = torch.ones_like(test_feature[0], requires_grad=True).to(args.device)
    beta = torch.zeros_like(test_feature[0], requires_grad=True).to(args.device)

    optimizer = torch.optim.AdamW([gamma, beta], lr=0.1)

    for epoch in range(10):
        out = model.recon_head(test_feature * gamma + beta)
        # loss = torch.mean(torch.pow(recon_out_single - test_sample, 2))

        orig_loss = F.mse_loss(out, test_x, reduction='none').mean(0)

        # give more weights to columns with higher loss within mse
        detached_loss = orig_loss.detach()
        normalized_detached_loss = F.normalize(detached_loss, p=1, dim=0)
        loss = orig_loss * normalized_detached_loss * torch.sum(detached_loss)

        optimizer.zero_grad()
        loss.mean().backward()
        print(f'epoch {epoch} loss is {orig_loss[7]}')
        optimizer.step()

    print(f'gamma - mse is {gamma}')
    print(f'beta - mse is {beta}')
    print('\n')

    # do = nn.Dropout(0.75)
    test_feature = model.get_feature(test_x).detach()


    # use the gamma and beta to reconstruct the test sample
    out = model.main_head(test_feature * gamma + beta)
    pred_list.append(out)

    pred_list = torch.stack(pred_list)
    test_acc = (torch.argmax(pred_list, dim=-1) == torch.argmax(test_y, dim=-1)).float()
    print(f'test acc - mse is {torch.mean(test_acc)}')

def mmd_loss(args, dataset, model):
    import torch

    def gaussian_kernel(x, y, sigma=1.0):
        """Computes a Gaussian kernel between x and y with a given sigma."""
        beta = 1.0 / (2.0 * sigma ** 2)
        dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2, p=2) ** 2
        s = torch.exp(-beta * dist)
        return s

    def mmd_loss(source_features, target_features, sigma=1.0):
        """Computes the MMD loss between source and target features using Gaussian kernels."""
        # Compute the kernels between source and target features
        xx_kernel = gaussian_kernel(source_features, source_features, sigma)
        yy_kernel = gaussian_kernel(target_features, target_features, sigma)
        xy_kernel = gaussian_kernel(source_features, target_features, sigma)

        # Compute the MMD loss
        xx_mean = torch.mean(xx_kernel)
        yy_mean = torch.mean(yy_kernel)
        xy_mean = torch.mean(xy_kernel)

        mmd = xx_mean - 2 * xy_mean + yy_mean
        return mmd

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    model.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


    for epoch in range(100):
        random_selection = torch.randperm(len(train_x))[:1024]
        source_features = model.get_feature(train_x[random_selection])
        target_features = model.get_feature(test_x[random_selection])

        optimizer.zero_grad()
        loss = mmd_loss(source_features, target_features, sigma=0.3)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # accuracy
            out = model(test_x)
            pred = torch.argmax(out, dim=-1)
            acc = torch.mean((pred == torch.argmax(test_y, dim=-1)).float())
            print(f'epoch {epoch} loss is {loss}, acc is {acc}')
    # print(loss)


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(args):
    from main import get_model
    logger = get_logger(args)
    device = args.device
    dataset = Dataset(args, logger)

    init_model = get_model(args, dataset)
    orig_model = get_model(args, dataset)
    dir_path = './'
    # print(os.listdir('../'))# get initalized model architecture only
    if os.path.exists(os.path.join(dir_path, args.out_dir, "source_model.pth")):
        init_model.load_state_dict(torch.load(os.path.join(dir_path, args.out_dir, "source_model.pth")))
        source_model = init_model
        source_model.eval()
        source_model.requires_grad_(False)
        source_model.to(args.device)
    else:
        raise FileNotFoundError

    orig_model.train()
    orig_model.requires_grad_(True)
    orig_model.to(device)

    train_x, train_y = torch.tensor(dataset.train_x).float().to(args.device), torch.tensor(dataset.train_y).float().to(
        args.device)
    test_x, test_y = torch.tensor(dataset.test_x).float().to(args.device), torch.tensor(dataset.test_y).float().to(
        args.device)
    optimizer = torch.optim.AdamW(orig_model.parameters(), lr=0.01)

    # test_cont_rescaled = dataset.test_cont_x_scaled
    # # replace the first n columns with test_rescaled
    # test_x_rescaled = test_x.clone().float().to(args.device)
    # test_x_rescaled[:, :test_cont_rescaled.shape[1]] = torch.tensor(test_cont_rescaled).float().to(args.device)

    for epoch in range(20):
        estim_y = orig_model(test_x)
        optimizer.zero_grad()
        F.cross_entropy(estim_y, torch.argmax(test_y, dim=-1)).mean().backward()
        optimizer.step()

        with torch.no_grad():
            estim_y = orig_model(test_x)
            # print(f"epoch {epoch}, loss {F.cross_entropy(estim_y, torch.argmax(test_y, dim=-1)).mean().item()}")
            print(f"epoch {epoch}, acc {(torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item()}")

    test_model = orig_model

    with torch.no_grad():
        estim_y = source_model(test_x)
        print(f"epoch {epoch}, acc {(torch.argmax(estim_y, dim=-1) == torch.argmax(test_y, dim=-1)).float().mean().item()}")
    # plot_mean_std_column(args, dataset)
    # plot_feature_output(args, dataset, source_model, test_model)
    # plot_recon_feature(args, dataset, source_model)
    # plot_acc_over_relative_entropy(args, dataset, source_model)
    # plot_stunt_tasks(args, dataset, source_model)
    # plot_maskratio(args, dataset, source_model)
    # pooled_avg_plot(args, source_model, dataset)
    # plot_recon_loss_per_column(args, source_model, dataset)
    # plot_recon_loss_per_column_per_class(args, source_model, dataset)
    # plot_shap_per_column(args, dataset, source_model, test_model)
    # prediction_based_on_mask(args, source_model, dataset)
    # linear_layer_exp(args, source_model, dataset)
    # interpolation(args, source_model, dataset)

    # plot_learned_bias(args, source_model, dataset)
    # plot_learned_bias(args, test_model, dataset)
    # cont_cat_features(args, source_model, dataset)
    # conv_features(args, dataset, source_model)
    # plot_permutation_result(args, dataset, source_model)
    # permutation_invariant(args, dataset, source_model, test_model)
    # plot_gumbel_softmax(args,dataset, source_model)
    # permuation_invariant(args, dataset, source_model)
    # use_avg_feature(args, dataset, source_model)
    # smotish(args, dataset, source_model)
    # give_adaptive_recon_loss(args, dataset, source_model, test_model)
    # subtract_emd(args, source_model, dataset)
    # plot_distance_over_acc(args, dataset, source_model)
    recon_wise_correction(args, dataset, source_model)
    # mmd_loss(args, dataset, source_model)

if __name__ == "__main__":
    set_seed(0)
    print(f"set seed as 0")
    main()