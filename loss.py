import numpy as np
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'BoundaryLoss'

    def forward(self, student_pred, teacher_pred, ground_truth, boundary=0.01, mask=None, interpolate=True):
        if interpolate:
            student_pred = nn.functional.interpolate(student_pred, ground_truth.shape[-2:], mode='bilinear', align_corners=True)
            teacher_pred = nn.functional.interpolate(teacher_pred, ground_truth.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            student_pred = student_pred[mask]
            teacher_pred = teacher_pred[mask]
            ground_truth = ground_truth[mask]

        student_log_error = torch.pow(torch.log(student_pred + 1e-9) - torch.log(ground_truth), 2)
        teacher_log_error = torch.pow(torch.log(teacher_pred + 1e-9) - torch.log(ground_truth), 2)

        student_log_error = torch.sum(student_log_error[student_log_error + boundary > teacher_log_error])

        return student_log_error / ground_truth.numel()
    
class CenterDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "CenterDifferenceLoss"

    def forward(self, student_center, teacher_center):
        return nn.functional.mse_loss(student_center, teacher_center)
    
class ProbDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ProbDifferenceLoss"

    def forward(self, student_p, teacher_p):
        return nn.functional.cross_entropy(student_p, teacher_p)
    
class FeatureMapDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "FeatureMapDifferenceLoss"

    def forward(self, student_feature, teacher_feature):
        return nn.functional.mse_loss(student_feature, teacher_feature)