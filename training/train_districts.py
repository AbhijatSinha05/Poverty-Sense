import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torchvision.transforms as T
from models.fusion import PovertyModel
from utils.satellite import fetch_satellite
from utils.graph_builder import build_graph

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load district centroids + MPI
centroids = pd.read_csv("data/tamilnadu_district_centroids.csv")
mpi = pd.read_csv("data/tamilnadu_mpi.csv")
df = centroids.merge(mpi, on="district")

model = PovertyModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
criterion = torch.nn.SmoothL1Loss()   # ✅ Huber loss

# ---- Multi-tile sampling offsets ----
offsets = [(0,0),(0.02,0),(0,-0.02),(0.02,0.02)]

def train(epochs=15):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for row in df.itertuples():
            lat, lon, label = row.lat, row.lon, row.mpi
            
            # ---- Multi-sample ViT embedding ----
            vit_embeds = []
            for dx, dy in offsets:
                img = transform(fetch_satellite(lat+dx, lon+dy)).unsqueeze(0).to(device)
                with torch.no_grad():
                    vit_embeds.append(model.vision(img))
            
            vit_embed = torch.mean(torch.stack(vit_embeds), dim=0)

            # ---- Build graph ----
            graph = build_graph(lat, lon).to(device)

            # ---- Assign ViT mean as node features ----
            graph.x[:] = vit_embed.mean()

            target = torch.tensor([[label]], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred = model(img, graph.x, graph.edge_index)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(df):.4f}")

    torch.save(model.state_dict(), "povertysense_model.pth")
    print("✅ Model trained and saved")

if __name__ == "__main__":
    train()
