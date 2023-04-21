import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LeNet5(nn.Module):
    def __init__(self, opt):
        super().__init__()

        h, w = opt["img_size"]
        in_chans = opt["in_chans"]
        class_num = opt["class_num"]

        self.convs = nn.Sequential(
            nn.Conv2d(in_chans, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        cur_h = max(h // 4 - 3, 1)
        cur_w = max(w // 4 - 3, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(16 * cur_h * cur_w, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, class_num),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x