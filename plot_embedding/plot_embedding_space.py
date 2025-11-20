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

# print('mu,sig = ')
# print(mu, sig)


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

# plot and label each ORC
# size = 10 * marker_size
# for i, (x, y) in enumerate(orc_umap):
#     size += 10 * marker_size
#     ax.scatter(
#         x, y,
#         s=size,
#         marker="X",
#         color=f"C{i%10}",
#         label=f"ORC {i}",
#         alpha=0.3
#     )


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
