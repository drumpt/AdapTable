import torch
import torch.nn.functional as F


def cat_aware_recon_loss(estimated_x, gt_x, model, cat_loss='mae'): # cat_loss: 'mse', 'ce'
    if cat_loss == 'mse':
        recon_loss = F.mse_loss(estimated_x, gt_x)
    else:
        recon_loss = F.mse_loss(estimated_x[:, :model.cat_start_index], gt_x[:, :model.cat_start_index]) * (model.cat_start_index / estimated_x.shape[-1]) # numerical loss
        for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices): # categorical loss
            recon_loss += F.cross_entropy(
                estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                torch.argmax(gt_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx], dim=-1)
            ).mean() * ((cat_end_idx - cat_start_idx) / estimated_x.shape[-1])
    return recon_loss


def expand_mask(mask, model):
    copy_mask = mask[:model.cat_start_index].clone()
    mask_idx = model.cat_start_index
    for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
        copy_mask = torch.cat([copy_mask, mask[mask_idx].repeat(cat_end_idx - cat_start_idx)], dim=0)
        mask_idx += 1
    return copy_mask