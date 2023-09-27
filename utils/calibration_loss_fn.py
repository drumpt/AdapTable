import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import *

class Posttrain_loss(nn.Module):
    def __init__(self, shrinkage_factor, prob_dist):
        super(Posttrain_loss, self).__init__()
        self.shrinkage_factor = shrinkage_factor

        # alpha calculation
        self.alpha = F.normalize(torch.reciprocal(prob_dist + 1e-6), p=1, dim=0)
        self.main_loss = FocalLoss(gamma=2, alpha=self.alpha)
        self.aux_loss = CAGCN_loss()

    def forward(self, logits, gt_label):
        # probs are logtis
        # gt_label are one-hot vectors
        # if gt_label.dim() == 1:
        #     gt_label = torch.argmax(gt_label, dim=-1)

        # cross entropy loss
        main_loss = self.main_loss(logits, gt_label)

        # CAGCN loss
        aux_loss = self.aux_loss(F.softmax(logits, dim=1), gt_label)

        # total loss
        loss = main_loss + self.shrinkage_factor * aux_loss
        return loss

class CAGCN_entropy(nn.Module):
    def __init__(self):
        super(CAGCN_entropy, self).__init__()

    def forward(self, probs, gt_label):
        correct_idx = (torch.argmax(probs, dim=-1) == torch.argmax(gt_label, dim=-1))
        incorrect_idx = (torch.argmax(probs, dim=-1) != torch.argmax(gt_label, dim=-1))

        correct_entropy = softmax_entropy(probs[correct_idx]).sum()
        incorrect_entropy = softmax_entropy(probs[incorrect_idx]).sum()

        loss = (correct_entropy - incorrect_entropy)/2
        return loss




class CAGCN_loss(nn.Module):
    def __init__(self):
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
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        if target.dim()==2:
            target = torch.argmax(target, dim=-1)
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()