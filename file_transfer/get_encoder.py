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

torch.save(encoder.state_dict(), "encoder.pt")