import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

pred = pd.read_csv("tamilnadu_districts_poverty_vit.csv")
gt = pd.read_csv("data/tamilnadu_mpi.csv")

df = pred.merge(gt, on="district")

df["poverty_norm"] = (df["poverty_index"] - df["poverty_index"].min()) / (df["poverty_index"].max() - df["poverty_index"].min())
df["mpi_norm"] = (df["mpi"] - df["mpi"].min()) / (df["mpi"].max() - df["mpi"].min())

spearman = spearmanr(df["poverty_norm"], df["mpi_norm"])[0]
print("Spearman rank correlation:", round(spearman,3))

df["pred_class"] = pd.qcut(df["poverty_norm"], 3, labels=["Low","Mid","High"], duplicates="drop")
df["mpi_class"] = pd.qcut(df["mpi_norm"], 3, labels=["Low","Mid","High"], duplicates="drop")

agreement = (df["pred_class"] == df["mpi_class"]).mean()
f1 = f1_score(df["mpi_class"], df["pred_class"], average="weighted")

print("Class agreement:", round(agreement*100,1),"%")
print("F1-score:", round(f1,3))
