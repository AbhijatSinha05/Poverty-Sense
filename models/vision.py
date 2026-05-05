import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

    def forward(self, x):
        return self.vit(x)
