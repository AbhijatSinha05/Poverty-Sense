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

centroids = pd.read_csv("data/tamilnadu_district_centroids.csv")

model = PovertyModel().to(device)
model.load_state_dict(torch.load("povertysense_model.pth", map_location=device))
model.eval()

results = []

for row in centroids.itertuples():
    lat, lon, name = row.lat, row.lon, row.district
    
    img = transform(fetch_satellite(lat, lon)).unsqueeze(0).to(device)
    graph = build_graph(lat, lon).to(device)

    # ViT embedding
    with torch.no_grad():
        vit_embed = model.vision(img)

    graph.x[:] = vit_embed.mean()

    with torch.no_grad():
        pred = model(img, graph.x, graph.edge_index)

    results.append([name, float(pred.cpu())])

df = pd.DataFrame(results, columns=["district","poverty_index"])
df.to_csv("tamilnadu_districts_poverty_vit.csv", index=False)

print("✅ District predictions saved")
