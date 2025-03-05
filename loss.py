"""loss functions."""
import torch


EPSILON = 1e-2


def _tonemap(im):
    im = torch.clamp(im, min=0)
    return im / (1+im)


def L1Loss(prediction, target):
    l1_loss = torch.nn.L1Loss(reduction='mean')

    return l1_loss(prediction, target)
    

def relativeL1Loss(prediction, target):
    loss = torch.abs(prediction - target) / (
        torch.abs(target.detach()) + EPSILON)

    return torch.mean(loss)
    

def relativeL2Loss(prediction, target):
    loss = torch.square(prediction - target) / (
        torch.square(target.detach()) + EPSILON)

    return torch.mean(loss)


def SMAPE(prediction, target):
    loss = torch.abs(prediction-target) / (
        torch.abs(prediction.detach()) + torch.abs(target.detach()) + EPSILON)

    return torch.mean(loss)


def tonemappedRelativeMSE(prediction, target):
    prediction = _tonemap(prediction)
    target = _tonemap(target)
    loss = torch.square(prediction - target) / (
        torch.square(target.detach() + EPSILON))
    
    return torch.mean(loss)
