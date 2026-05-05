import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load predictions
# -----------------------------
pred = pd.read_csv("tamilnadu_districts_poverty_gnn.csv")

# -----------------------------
# Load GeoJSON
# -----------------------------
geo = gpd.read_file("data/TamilNadu.geojson")

# Rename correct district column
geo = geo.rename(columns={"NAME_2": "district"})

geo["district"] = geo["district"].str.strip()
pred["district"] = pred["district"].str.strip()

# Merge spatial + prediction data
gdf = geo.merge(pred, on="district")

# -----------------------------
# Create 3 poverty classes
# -----------------------------
gdf["poverty_class"] = pd.qcut(
    gdf["poverty_index"],
    3,
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# Plot categorical heatmap
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8,10))

gdf.plot(
    column="poverty_class",
    cmap="RdYlGn_r",   # Green = Low, Red = High
    linewidth=0.8,
    ax=ax,
    edgecolor="black",
    legend=True
)

ax.set_title("District-level Poverty Classification (Tamil Nadu)", fontsize=14)
ax.axis("off")

# -----------------------------
# Save high-resolution PNG
# -----------------------------
plt.savefig("poverty_heatmap_3class.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Saved poverty_heatmap_3class.png")
