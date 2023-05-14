import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_augmentation(x, args):
    # MEMO with dropout
    dropout = nn.Dropout(p=0.1)
    x_aug = torch.stack([dropout(x) for _ in range(args.memo_aug_num - 1)]).to(args.device)
    return x_aug