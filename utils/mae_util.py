import torch
import torch.nn.functional as F


def cat_aware_recon_loss(estimated_x, orig_x, model, reduction='mean', cat_loss='mse'): # cat_loss: 'mse', 'ce'
    cont_part_recon = estimated_x[:, :model.cat_start_index]
    cont_part_orig = orig_x[:, :model.cat_start_index]
    cat_loss_fn = F.cross_entropy if cat_loss == 'ce' else F.mse_loss

    if reduction == 'none':
        recon_loss = F.mse_loss(cont_part_recon, cont_part_orig, reduction=reduction).mean(0)
        for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
            recon_loss = torch.cat([recon_loss, cat_loss_fn(
                estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                torch.argmax(orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx], dim=1) if cat_loss == 'ce' else orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                reduction=reduction,
            ).mean().reshape(-1)])
    else:
        recon_loss = F.mse_loss(cont_part_recon, cont_part_orig, reduction=reduction) * (model.cat_start_index / estimated_x.shape[-1])
        for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
            recon_loss += cat_loss_fn(
                estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                torch.argmax(orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx], dim=1) if cat_loss == 'ce' else orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                reduction=reduction,
            ) * ((cat_end_idx - cat_start_idx) / estimated_x.shape[-1])
    return recon_loss


def expand_mask(mask, model):
    copy_mask = mask[:model.cat_start_index].clone()
    mask_idx = model.cat_start_index
    for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
        copy_mask = torch.cat([copy_mask, mask[mask_idx].repeat(cat_end_idx - cat_start_idx)], dim=0)
        mask_idx += 1
    return copy_mask