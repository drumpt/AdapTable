import torch
import torch.nn.functional as F


def cat_aware_recon_loss(estimated_x, orig_x, model, reduction='mean'):
    cont_part_recon = estimated_x[:, :model.cat_start_index]
    cont_part_orig = orig_x[:, :model.cat_start_index]

    if reduction == 'none':
        recon_loss = F.mse_loss(cont_part_recon, cont_part_orig, reduction='none').mean(0)
        for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
            # print(f'pred : {estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx]}')
            # print(f'target : {orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx]}')
            recon_loss = torch.cat([recon_loss, (F.cross_entropy(
                estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                torch.argmax(orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx], dim=1),
            ).mean() / (cat_end_idx - cat_start_idx)).reshape(1)], dim=0)
    else:
        recon_loss = F.mse_loss(cont_part_recon, cont_part_orig).mean()
        for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
            recon_loss += F.cross_entropy(
                estimated_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx],
                torch.argmax(orig_x[:, model.cat_start_index + cat_start_idx:model.cat_start_index + cat_end_idx], dim=1)
            ).mean() / (cat_end_idx - cat_start_idx)
    return recon_loss

def expand_mask(mask, model):
    copy_mask = mask[:model.cat_start_index].clone()
    mask_idx = model.cat_start_index
    for cat_start_idx, cat_end_idx in zip(model.cat_start_indices, model.cat_end_indices):
        copy_mask = torch.cat([copy_mask, mask[mask_idx].repeat(cat_end_idx - cat_start_idx)], dim=0)
        mask_idx += 1
    return copy_mask