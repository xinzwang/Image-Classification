import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LiteNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        in_chans = opt["in_chans"]
        class_num = opt["class_num"]

        self.convs = nn.Sequential(
            nn.Conv2d(in_chans, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),

            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(32, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, class_num)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x