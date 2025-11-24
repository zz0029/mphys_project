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
pca_cols = [f"pca_{i}" for i in range(200)]
pca_df = pd.DataFrame(pca_feats, columns=pca_cols)

# 把必要信息加进去（顺序与原始 pca.parquet 一致：pca_*，rgz_name，size，ra，dec）
for col in ["rgz_name", "size", "ra", "dec"]:
    if col in df.columns:
        pca_df[col] = df[col].values

# 明确列顺序
ordered_pca_cols = pca_cols + [c for c in ["rgz_name", "size", "ra", "dec"] if c in pca_df.columns]
pca_df = pca_df[ordered_pca_cols]

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

# 保存为 umap.parquet：列顺序与原始 umap.parquet 一致：rgz_name, umap_x, umap_y, ra, dec, size
data = {}

# 先加元数据
for col in ["rgz_name"]:
    if col in df.columns:
        data[col] = df[col].values

# 再加 UMAP 坐标
data["umap_x"] = umap_xy[:, 0]
data["umap_y"] = umap_xy[:, 1]

# 最后加位置信息和 size（如果存在）
for col in ["ra", "dec", "size"]:
    if col in df.columns:
        data[col] = df[col].values

umap_df = pd.DataFrame(data)

umap_df.to_parquet("umap.parquet")
print("Saved umap.parquet")