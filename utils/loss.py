import torch
import torch.nn as nn


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, pred, target, mask, eos_idx):

        pred = to_contiguous(pred).view(-1, pred.size(-1))
        target = torch.cat([target[:, 1:], target[:, 0].unsqueeze(1).fill_(eos_idx)], dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -1. * pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output.float()


class SoftCriterion(nn.Module):
    def __init__(self):
        super(SoftCriterion, self).__init__()

    def forward(self, pred, idxs, soft_target, mask):
        topk = -1.0 * pred.gather(-1, idxs) * mask[..., None]
        output = soft_target * topk
        output = torch.sum(output) / torch.sum(mask)
        return output.float()


class CosineCriterion(nn.Module):
    def __init__(self):
        super(CosineCriterion, self).__init__()
        self.eps = 1e-12

    def forward(self, pred, target):
        assert pred.shape == target.shape and pred.dim() == 2, \
            'expected pred.shape == target.shape, ' \
            'but got pred.shape == {} and target.shape == {}'.format(pred.shape, target.shape)
        pred_denom = torch.norm(pred, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(pred)
        pred = pred / pred_denom
        target_denom = torch.norm(target, p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(target)
        target = target / target_denom

        ret = pred * target
        ret = 1.0 - ret.sum(dim=-1)
        ret = ret.sum()
        return ret



