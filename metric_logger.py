import torch
from piq import psnr, ssim
from loss import L1Loss, SMAPE
import numpy as np

def _tonemap(img, gamma):
    return img.clamp(0, 1)

class MetricLogger:
    def __init__(self, gamma=2.4):
        self.gamma = gamma
        self.reset_metrics()

    def reset_metrics(self):
        self.metrics = {metric: [] for metric in ["L1", "SMAPE", "PSNR", "SSIM"]}

    def add(self, prediction, gt):
        with torch.no_grad():
            assert torch.isfinite(prediction).all()

            ldr_prediction = _tonemap(prediction, self.gamma)
            ldr_gt = _tonemap(gt, self.gamma)

            iteration_info = {
                "L1": L1Loss(prediction, gt).item(),
                "SMAPE": SMAPE(prediction, gt).item(),
                "PSNR": psnr(ldr_prediction, ldr_gt, data_range=1.0, reduction='mean').item(),
                "SSIM": ssim(ldr_prediction, ldr_gt, data_range=1.0, reduction='mean').item(),
            }

            for key, value in iteration_info.items():
                self.metrics[key].append(value)

            return iteration_info

    def getEpochInfo(self):
        return {
            key: {
                "average": float(np.mean(values)),
                "std_err": float(np.std(values)/np.sqrt(len(values)))
            } for key, values in self.metrics.items()
        }