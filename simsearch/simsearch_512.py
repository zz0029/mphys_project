import os

os.environ["MKL_THREADING_LAYER"] = "TBB"

import pylab as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

from byol.datasets import RGZ108k
from byol.datasets import MBFRFull
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
        """Read pre-computed 512D embeddings from parquet file."""
        print(f"Reading embedding from file: {self.filename}")
        df = pd.read_parquet(self.filename)
        features = df[[f"feat_{i}" for i in range(512)]].values
        if "target" in df.columns:
            targets = df["target"].values
        else:
            targets = np.ones(features.shape[0])
        return features, targets

    def write_file(self, filename):
        """Write 512D embeddings to parquet file for later reuse."""
        cols = [f"feat_{i}" for i in range(512)]
        print(self.features.shape, self.targets.shape)
        df = pd.DataFrame(data=self.features, columns=cols)
        df.to_parquet(filename)
        return

    def embed_dataset(self, data, batch_size=400):
        """Pass an entire Dataset through the encoder to get 512D features."""
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(self.encoder.parameters()).device
        feature_bank = []

        for batch in tqdm(train_loader):
            # RGZ108k returns (image, meta_dict); TensorDataset returns (image, dummy_label)
            if len(batch) > 2:
                x = batch
            else:
                x, _ = batch

            with torch.no_grad():
                x_enc = self.encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().cpu())

        features = torch.cat(feature_bank)
        targets = np.ones(features.shape[0])
        return features, targets

    def fit(self, data=None):
        """Fit PCA + UMAP using 512D BYOL features."""
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

        # Ensure we have a numpy array for sklearn
        if isinstance(self.features, torch.Tensor):
            feat_np = self.features.numpy()
        else:
            feat_np = self.features

        self.pca.fit(feat_np)
        self.umap.fit(self.pca.transform(feat_np))
        return

    def transform(self, data=None):
        """Apply PCA+UMAP to either new data or previously stored features."""
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

        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = x

        x_np = self.pca.transform(x_np)
        x_np = self.umap.transform(x_np)
        return x_np

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = x
        x_np = self.pca.transform(x_np)
        return x_np


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
orc_pt_path = "/share/nas2_3/yhuang/myrepo/mphys_project/preprocess_orc/orc_cutouts.pt"
orc_data = torch.load(orc_pt_path)

orc_images = orc_data["images"].float()  # shape: (N, 1, 70, 70)
orc_labels = orc_data["labels"]

# make a Dataset so Reducer.embed_dataset can read it
dummy_targets = torch.zeros(len(orc_images))  # placeholder labels
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


# ================== UMAP embedding for plotting ==================
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


# ============================== Plot RGZ + MB + ORC ==========================
# PCA + UMAP
rgz_umap = reducer.transform()
mb_umap = reducer.transform(mb)
orc_umap = reducer.transform(orc_dataset)

# print ORC positions
print("ORC UMAP positions:")
for i, (x, y_umap) in enumerate(orc_umap):
    print(i, x, y_umap)

# Plotting
fig, ax = pl.subplots()
ax.scatter(rgz_umap[:, 0], rgz_umap[:, 1], label="RGZ DR1", marker=marker, s=marker_size, alpha=alpha)
ax.scatter(mb_umap[:, 0], mb_umap[:, 1], label="MiraBest", marker=marker, s=10 * marker_size, alpha=alpha)
ax.scatter(orc_umap[:, 0], orc_umap[:, 1], label="ORC", marker="x", s=100 * marker_size, alpha=alpha, c="red")

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


# ============================== Similarity Search in 512D ====================
import torchvision.utils as vutils

# ---- Get 512D features for RGZ (from reducer.fit) ----
if isinstance(reducer.features, torch.Tensor):
    rgz_features_512 = reducer.features.numpy()
else:
    rgz_features_512 = reducer.features

# ---- Get 512D features for ORCs (encode again using the same encoder) ----
orc_features_512, _ = reducer.embed_dataset(orc_dataset)
if isinstance(orc_features_512, torch.Tensor):
    orc_features_512 = orc_features_512.numpy()

# ---- Nearest Neighbours in 512D BYOL space using cosine distance ----
K = 5  # number of neighbours per ORC
nn = NearestNeighbors(n_neighbors=K, metric="cosine")
nn.fit(rgz_features_512)

# distances are actually cosine distances = 1 - cosine_similarity
distances, indices = nn.kneighbors(orc_features_512)

# ---- Save neighbour images and print information ----
out_dir = "orc_nearest_rgz_512_cosine"
os.makedirs(out_dir, exist_ok=True)

print("===== ORC 最近 RGZ 邻居（在 512D BYOL 空间，cosine similarity）=====")

for orc_idx, (dists, idxs) in enumerate(zip(distances, indices)):
    print(f"\nORC {orc_idx}:")

    for rank, (dist, rgz_idx) in enumerate(zip(dists, idxs), start=1):
        cos_sim = 1.0 - dist
        umap_xy = rgz_umap[rgz_idx]
        print(
            f"  #{rank}: RGZ index {rgz_idx}, "
            f"UMAP=({umap_xy[0]:.4f}, {umap_xy[1]:.4f}), "
            f"cosine_sim={cos_sim:.4f}"
        )

        # 从 Dataset 里取图像
        img, meta = rgz[rgz_idx]  # img: Tensor [1, H, W]

        # ---- 为了能看清细节，对图像做反归一化 + 按每张图单独拉伸到 [0,1] ----
        # 当前 img 已经经过 Normalize((mu,), (sig,))，大部分像素在 [-几, 几] 之间，
        # 直接保存会被截断成全黑。所以这里先还原，再做 min-max 归一化。
        img_vis = img.clone()

        # 反归一化：x_orig = x_norm * sig + mu
        img_vis = img_vis * sig + mu

        # 按每张图的 min / max 拉伸到 [0,1]
        min_val = img_vis.min()
        max_val = img_vis.max()
        img_vis = (img_vis - min_val) / (max_val - min_val + 1e-8)

        save_path = os.path.join(
            out_dir,
            f"orc{orc_idx}_nbr{rank}_rgz{rgz_idx}.png",
        )
        vutils.save_image(img_vis, save_path)

        # 打印 meta 信息，方便之后在 catalog 中定位
        print("    META for this RGZ sample:")
        try:
            for k, v in meta.items():
                # 对于标量或长度为1的张量，打印数值；否则打印形状
                if hasattr(v, "shape") and v.shape == ():
                    val_str = f"value={v.item()}"
                else:
                    val_str = f"shape={getattr(v, 'shape', 'N/A')}"
                print(f"      {k}: {val_str}")
        except Exception as e:
            print(f"      (Could not unpack meta dict cleanly: {e})")

print("\nDone: neighbour images saved in", out_dir)