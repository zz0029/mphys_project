import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
from umap import UMAP

# ========================================================
# 你的 parquet 文件（512维 BYOL 特征）
# ========================================================
parquet_path = "/share/nas2_3/yhuang/_data/rgz/rgz_embedding_25.parquet"

df = pd.read_parquet(parquet_path)
features = df[[f"feat_{i}" for i in range(512)]].values

# ========================================================
# PCA → 200维
# ========================================================
print("Fitting PCA (200 dims)...")
pca = PCA(n_components=200, random_state=42)
pca_feats = pca.fit_transform(features)

# 保存为 latent explorer 用的 pca.joblib
joblib.dump(pca, "pca.joblib")
print("Saved pca.joblib")

# 保存 200维特征到 pca.parquet
pca_df = pd.DataFrame({
    f"pca_{i}": pca_feats[:, i] for i in range(200)
})

# 把必要信息加进去（RA/Dec/name）
for col in ["ra", "dec", "rgz_name"]:
    if col in df.columns:
        pca_df[col] = df[col]

pca_df.to_parquet("pca.parquet")
print("Saved pca.parquet")

# ========================================================
# UMAP → 2维
# ========================================================
print("Fitting UMAP (2 dims)...")
umap_model = UMAP(
    n_components=2,
    n_neighbors=75,     # keep same as your plot_embedding_space.py
    min_dist=0.01,
    metric="cosine",
    random_state=42,
)

umap_xy = umap_model.fit_transform(pca_feats)

# 保存为 umap.parquet
umap_df = pd.DataFrame({
    "umap_x": umap_xy[:, 0],
    "umap_y": umap_xy[:, 1]
})
umap_df.to_parquet("umap.parquet")
print("Saved umap.parquet")