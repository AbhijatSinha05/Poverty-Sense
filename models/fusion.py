import torch
import torch.nn as nn
from models.vision import VisionEncoder
from models.graph import RoadGCN

class PovertyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionEncoder()
        self.graph = RoadGCN()
        
        # Fusion head
        self.fc = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, node_x, edge_index):
        v = self.vision(image)              # [1,768]
        g = self.graph(node_x, edge_index)  # [N,64]
        g = g.mean(dim=0, keepdim=True)     # [1,64]
        fused = torch.cat([v, g], dim=1)    # [1,832]
        return self.fc(fused)               # ⚠️ No sigmoid
