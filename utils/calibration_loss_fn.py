import torch
import torch.nn as nn
import torch.nn.functional as F



class CAGCN_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(CAGCN_loss, self).__init__()


    def forward(self, probs, gt_label):
        correct_idx = (torch.argmax(probs, dim=-1) == torch.argmax(gt_label, dim=-1))
        incorrect_idx = (torch.argmax(probs, dim=-1) != torch.argmax(gt_label, dim=-1))

        top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
        largest_val = top2_probs[:, 0]
        second_largest_val = top2_probs[:, 1]

        correct_loss = 1 / probs.shape[1] * (torch.sum(
            1 - largest_val[correct_idx] + second_largest_val[correct_idx])
        )
        incorrect_loss = 1 / probs.shape[1] * (torch.sum(
            largest_val[incorrect_idx] - second_largest_val[incorrect_idx])
        )
        loss = correct_loss + incorrect_loss
        loss /= len(probs)
        return loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, inputs, targets):
        # Compute the cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none').unsqueeze(1)

        # Get the probabilities for the true class
        probs = torch.exp(-ce_loss).unsqueeze(1)

        # Compute the focal loss
        focal_loss = (torch.mul(self.alpha.repeat(len(probs), 1), (1 - probs)) ** self.gamma) * ce_loss

        # Reduce the loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss