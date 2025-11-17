import os
os.environ["MKL_THREADING_LAYER"] = "TBB"

import pylab as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


from sklearn.decomposition import PCA
from umap import UMAP


from byol.utilities import embed_dataset
from byol.datasets import RGZ108k
from byol.datasets import MBFRFull, MBHybrid, MBFRConfident, MBFRUncertain
from byol.models import BYOL

#Define location of checkpoint file for pre-trained BYOL model:
ckpt = '/share/nas2_3/yhuang/byol/runscripts/byol.ckpt'

#Load model from checkpoint (pre-trained weights):
byol = BYOL.load_from_checkpoint(ckpt)
byol.eval()
encoder = byol.encoder
encoder.eval()
config = byol.config
mu, sig = config["data"]["mu"], config["data"]["sig"]

#Define a class of functions to encode the data through the model and perform dimensionality reduction:
class Reducer:

    def __init__(self, encoder, PCA_COMPONENTS, UMAP_N_NEIGHBOURS, UMAP_MIN_DIST, METRIC, embedding=None, seed=42):

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
                print("Specified embedding file does not exist - will compute embedding")
                self.embedded = False
            else:
                self.filename = embedding
                self.embedded = True
        else:
            self.embedded = False

    def read_file(self):

        print("Reading embedding from file: {}".format(self.filename))

        df = pd.read_parquet(self.filename)
        features = df[[f"feat_{i}" for i in range(512)]].values
        if 'target' in df.columns:
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
        device = next(encoder.parameters()).device
        feature_bank = []
        target_bank = []
        for data in tqdm(train_loader):
            # print(len(data), data[0].shape)
            # Load data and move to correct device
            if len(data) > 2:
                x = data
            else:
                x, y = data

            x_enc = encoder(x.to(device))

            feature_bank.append(x_enc.squeeze().detach().cpu())
            # target_bank.append(y['size'].detach().cpu())

        # Save full feature bank for validation epoch
        features = torch.cat(feature_bank)
        # targets = torch.cat(target_bank)
        targets = np.ones(features.shape[0])

        return features, targets

    def fit(self, data=None):

        print("Fitting reducer")

        if data != None: features, targets = self.embed_dataset(data)
        if data == None and self.embedded: features, targets = self.read_file()
        if data == None and not self.embedded:
            print("No data/embedding provided - exiting")
            return

        self.features = features
        self.targets = targets

        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))

        return

    def transform(self, data=None):

        print("Performing transformation")

        if data != None:
            x, _ = self.embed_dataset(data)
        elif data == None and hasattr(self, 'features'):
            x = self.features
        elif data == None and not hasattr(self, 'features') and self.embedded:
            x, _ = self.read_file()
        elif data == None and not hasattr(self, 'features') and not self.embedded:
            print("No data/embedding provided - exiting")
            return

        x = self.pca.transform(x)
        x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x

# Load the datasets (RGZ and MiraBest):
paths={}
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
    cut_threshold=25,           # remove sources below this size threshold [arcsec]
    mb_cut=True,                # remove sources that are also in MiraBest
    )

mb = MBFRFull(paths["mb"],
              train=True,
              transform=transform,
              download=False,
              aug_type="torchvision"
             )