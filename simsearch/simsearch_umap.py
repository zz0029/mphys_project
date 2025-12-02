import os

os.environ["MKL_THREADING_LAYER"] = "TBB"

import pylab as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from umap import UMAP
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

from byol.datasets import RGZ108k
from byol.datasets import MBFRFull, MBHybrid, MBFRConfident, MBFRUncertain
from byol.models import BYOL

# ================== BYOL checkpoint ==================
ckpt = "/share/nas2_3/yhuang/byol/runscripts/byol.ckpt"

# Load model from checkpoint (pre-trained weights):
byol = BYOL.load_from_checkpoint(ckpt)
byol.eval()
encoder = byol.encoder
encoder.eval()
config = byol.config
mu, sig = config["data"]["mu"], config["data"]["sig"]

# print(mu, type(mu))
# print(sig, type(sig))


# ================== Reducer class ==================
class Reducer:
    def __init__(
            self,
            encoder,
            PCA_COMPONENTS,
            UMAP_N_NEIGHBOURS,
            UMAP_MIN_DIST,
            METRIC,
            embedding=None,
            seed=42,
    ):
        self.encoder = encoder
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        self.umap = UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBOURS,
            min_dist=UMAP_MIN_DIST,
            metric=METRIC,
            random_state=seed,
        )

        if embedding is not None:
            if not os.path.exists(embedding):
                print(
                    "Specified embedding file does not exist - will compute embedding"
                )
                self.embedded = False
            else:
                self.filename = embedding
                self.embedded = True
        else:
            self.embedded = False

    def read_file(self):
        print(f"Reading embedding from file: {self.filename}")
        df = pd.read_parquet(self.filename)
        features = df[[f"feat_{i}" for i in range(512)]].values
        if "target" in df.columns:
            targets = df["target"].values
        else:
            targets = np.ones(features.shape[0])
        return features, targets

    def write_file(self, filename):
        cols = [f"feat_{i}" for i in range(512)]
        print(self.features.shape, self.targets.shape)
        df = pd.DataFrame(data=self.features, columns=cols)
        df.to_parquet(filename)
        return

    def embed_dataset(self, data, batch_size=400):
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(self.encoder.parameters()).device
        feature_bank = []

        for batch in tqdm(train_loader):
            if len(batch) > 2:
                x = batch
            else:
                x, y = batch

            with torch.no_grad():
                x_enc = self.encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().cpu())

        features = torch.cat(feature_bank)
        targets = np.ones(features.shape[0])
        return features, targets

    def fit(self, data=None):
        print("Fitting reducer")

        if data is not None:
            features, targets = self.embed_dataset(data)
        elif data is None and self.embedded:
            features, targets = self.read_file()
        else:
            print("No data/embedding provided - exiting")
            return

        self.features = features
        self.targets = targets

        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))
        return

    def transform(self, data=None):
        print("Performing transformation")

        if data is not None:
            x, _ = self.embed_dataset(data)
        elif data is None and hasattr(self, "features"):
            x = self.features
        elif data is None and not hasattr(self, "features") and self.embedded:
            x, _ = self.read_file()
        else:
            print("No data/embedding provided - exiting")
            return

        x = self.pca.transform(x)
        x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x


# ================== Datasets ==================
paths = {}
paths["rgz"] = "/share/nas2_3/yhuang/_data/rgz"
paths["mb"] = "/share/nas2_3/yhuang/_data/mb"

transform = T.Compose(
    [
        T.CenterCrop(70),
        T.ToTensor(),
        T.Normalize((mu,), (sig,)),
    ]
)

rgz = RGZ108k(
    paths["rgz"],
    train=True,
    transform=transform,
    download=False,
    remove_duplicates=False,
    cut_threshold=25,  # remove sources below this size threshold [arcsec]
    mb_cut=True,  # remove sources that are also in MiraBest
)

mb = MBFRFull(
    paths["mb"],
    train=True,
    transform=transform,
    download=False,
    aug_type="torchvision",
)

# ================== ORC cutouts dataset ==================
from torch.utils.data import TensorDataset

orc_pt_path = "/share/nas2_3/yhuang/myrepo/mphys_project/preprocess_orc/orc_cutouts.pt"
orc_data = torch.load(orc_pt_path)

orc_images = orc_data["images"].float()  # shape: (N, 1, 70, 70)
orc_labels = orc_data["labels"]

# make a Dataset for Reducer.embed_dataset can read
dummy_targets = torch.zeros(len(orc_images))  # 占位 label
orc_dataset = TensorDataset(orc_images, dummy_targets)

# ================== Dimensionality reduction hyperparameters ==================
embedding_file = "/share/nas2_3/yhuang/_data/rgz/rgz_embedding_25.parquet"
PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 75
UMAP_MIN_DIST = 0.01
METRIC = "cosine"


reducer = Reducer(
    encoder,
    PCA_COMPONENTS,
    UMAP_N_NEIGHBOURS,
    UMAP_MIN_DIST,
    METRIC,
    embedding=embedding_file,
)

# ======= 自动判断：第一次运行 or 之后重复运行 =======
if os.path.exists(embedding_file):
    print("Embedding file found. Loading features from disk.")
    reducer.fit()  # 使用 parquet 中的特征
else:
    print("Embedding file NOT found. Computing features from RGZ (this may take a while).")
    reducer.fit(data=rgz)
    reducer.write_file(embedding_file)


X_umap = reducer.transform()

# ================== Plot (Figure 2 style) ==================
alpha = 0.6
marker_size = 0.1
fig_size = (10 / 3, 3)
fontsize = 9
marker = "o"

# Get the label info for all the embedded data samples:
data_loader = DataLoader(rgz, batch_size=len(rgz), shuffle=False)
_, y = next(iter(data_loader))

fig, ax = pl.subplots()
# fig.set_size_inches(fig_size)

scatter = ax.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=y["size"].numpy(),
    cmap="Spectral",
    s=marker_size,
    marker=marker,
    vmin=25,
    vmax=100,
    alpha=alpha,
)

pl.gca().set_aspect("equal", "datalim")
cbar = fig.colorbar(scatter)
cbar.ax.tick_params(labelsize=fontsize)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.savefig("byol_umap_rgz.png", bbox_inches="tight", pad_inches=0.05, dpi=600)

# ==============================Plot RGZ + MB + ORC ==========
# PCA + UMAP
rgz_umap = reducer.transform()
mb_umap = reducer.transform(mb)
orc_umap = reducer.transform(orc_dataset)

# print orc's posiions
print("ORC UMAP positions:")
for i, (x, y) in enumerate(orc_umap):
    print(i, x, y)

# Plotting
fig, ax = pl.subplots()
ax.scatter(rgz_umap[:, 0], rgz_umap[:, 1], label="RGZ DR1", marker=marker, s=marker_size, alpha=alpha)
ax.scatter(mb_umap[:, 0], mb_umap[:, 1], label="MiraBest", marker=marker, s=10 * marker_size, alpha=alpha)
ax.scatter(orc_umap[:, 0], orc_umap[:, 1], label="ORC", marker="x", s=100 * marker_size, alpha=alpha, c= "red")



# Labels on graph
ax.legend()
ax.set_xlabel("umap x", fontsize=fontsize)
pl.gca().set_aspect("equal", "datalim")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
ax.legend(fontsize=fontsize, markerscale=10)
fig.tight_layout()
fig.savefig("byol_umap_mbrgz_orc.png", bbox_inches="tight", pad_inches=0.05, dpi=600)


# ==============================Similarity Search==========
from sklearn.neighbors import NearestNeighbors
import os
import torchvision.utils as vutils

# ========= 在 UMAP 空间里做最近邻 =========
K = 5  # 每个 ORC 找 5 个最近的 RGZ
nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
nn.fit(rgz_umap)  # 只用 RGZ 的 UMAP 坐标建索引

distances, indices = nn.kneighbors(orc_umap)  # 对每个 ORC 求最近邻

# 把结果打印出来 + 把对应 RGZ 图像存成 PNG
out_dir = "orc_nearest_rgz"
os.makedirs(out_dir, exist_ok=True)

print("===== ORC 最近 RGZ 邻居（在 UMAP 2D 空间）=====")

for orc_idx, (dists, idxs) in enumerate(zip(distances, indices)):
    print(f"\nORC {orc_idx} at UMAP {orc_umap[orc_idx]}")

    for rank, (dist, rgz_idx) in enumerate(zip(dists, idxs), start=1):
        umap_xy = rgz_umap[rgz_idx]
        print(f"  #{rank}: RGZ index {rgz_idx}, "
              f"UMAP=({umap_xy[0]:.4f}, {umap_xy[1]:.4f}), "
              f"dist={dist:.4f}")

        # 从 Dataset 里取图像
        img, meta = rgz[rgz_idx]  # img: Tensor [1, H, W]

        # === DEBUG: print raw tensor + meta ===
        print("\nDEBUG: RAW RGZ IMAGE TENSOR")
        print("rgz_idx =", rgz_idx)
        print("shape =", img.shape)
        print("min   =", img.min().item())
        print("max   =", img.max().item())
        print("mean  =", img.mean().item())
        print("std   =", img.std().item())

        # 打印整张 image（注意会很长）
        print("full image tensor:")
        print(img)  # img 形状是 [1, H, W]
        # 或者只打掉 channel 这一维：
        # print(img[0])

        # 打印 meta 信息
        print("META for this RGZ sample:")
        print(meta)  # meta 是 RGZ108k 返回的字典

        # 如果你想更详细一点，可以这样展开：
        for k, v in meta.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, values={v}")

        print("============== END DEBUG ==============\n")

        # 可视化用的 tensor
        img_vis = img.clone()

        # 1. 反归一化：x_orig = x_norm * sig + mu  （撤销 BYOL 训练时的 Normalize）
        img_vis = img_vis * sig[0] + mu[0]   # [1, H, W]

        # 2. 转成 numpy 2D 数组，方便使用 astropy 的可视化工具
        img_np = img_vis.squeeze(0).cpu().numpy()   # [H, W]

        # 3. 计算 sigma-clipped 统计量（与 visualise_rgz.py 保持一致）
        mean, med, std = sigma_clipped_stats(img_np, sigma=3.0)

        # 4. 定义归一化：百分位截断 + asinh 拉伸
        norm = ImageNormalize(
            img_np,
            interval=PercentileInterval(99.5),  # 或 99.7，和你在 visualise_rgz.py 中保持一致
            stretch=AsinhStretch(),
        )

        # 5. 应用归一化，得到 [0,1] 的 2D 数组
        img_norm = norm(img_np).astype("float32")   # [H, W]

        # 6. 转回 torch.Tensor，并加回通道维度，以便 save_image 使用
        img_vis = torch.from_numpy(img_norm).unsqueeze(0)  # [1, H, W]

        save_path = os.path.join(
            out_dir,
            f"orc{orc_idx}_nbr{rank}_rgz{rgz_idx}.png"
        )
        vutils.save_image(img_vis, save_path)