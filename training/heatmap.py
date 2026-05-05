import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load your model predictions
# -----------------------------
# CSV must have columns: district, poverty_index
pred = pd.read_csv("tamilnadu_districts_poverty_vit.csv")

# -----------------------------
# Load Tamil Nadu district GeoJSON
# -----------------------------
geo = gpd.read_file("data/TamilNadu.geojson")
geo = geo.rename(columns={"NAME_2": "district"})
# -----------------------------
# Standardize district name column
# -----------------------------
geo["district"] = geo["district"].str.strip()
pred["district"] = pred["district"].str.strip()

# -----------------------------
# Merge spatial + prediction data
# -----------------------------
gdf = geo.merge(pred, on="district")

# -----------------------------
# Plot heatmap
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8,10))

gdf.plot(
    column="poverty_index",
    cmap="OrRd",
    linewidth=0.8,
    ax=ax,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Predicted Poverty Index"}
)

ax.set_title("District-level Poverty Heatmap (Tamil Nadu)", fontsize=14)
ax.axis("off")

# -----------------------------
# Save high-resolution PNG
# -----------------------------
plt.savefig("poverty_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Heatmap saved as poverty_heatmap.png")
