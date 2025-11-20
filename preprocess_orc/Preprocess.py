import torch
import numpy as np
from cata2data import CataData
import torchvision.transforms as T

# ========= 这里填写你的 9 个 fits 文件路径 =========
fits_paths = [
    # "../orc_fits/image.i.EMU_2034-60.SB72006.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_2042-55.SB72149.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_0205-55.SB46946.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_0354-55.SB59481.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_0448-60.SB50230.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_1315-46.SB51948.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_2236-69.SB53566.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_0857-04A.SB71464.cont.taylor.0.restored.conv.fits",
    "../orc_fits/image.i.EMU_2238-51.SB59862.cont.taylor.0.restored.conv.fits",
]

# ========= 对应 9 个 ORC 的坐标文件（每个文件 1 行）=========
catalogue_paths = [
    # "../orc_cata/ORC coordinates in 72006.txt",
    "../orc_cata/ORC coordinates in 72149.txt",
    "../orc_cata/ORC coordinates in 46946.txt",
    "../orc_cata/ORC coordinates in 59481.txt",
    "../orc_cata/ORC coordinates in 50230.txt",
    "../orc_cata/ORC coordinates in 51948.txt",
    "../orc_cata/ORC coordinates in 53566.txt",
    "../orc_cata/ORC coordinates in 71464.txt",
    "../orc_cata/ORC coordinates in 59862.txt",
]

assert len(fits_paths) == len(catalogue_paths), "fits 数量和坐标文件数量不一致！"

# ========= Transform，只做 Normalize（和 RGZ/MB 一致） =========
mu, sig = 0.008008896, 0.05303395    # 来自 byol.config["data"]["mu"], ["data"]["sig"]

# 注意：Normalize 只对 tensor 起作用，形状需为 [C,H,W]
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((mu,), (sig,)),
    ]
)

all_images = []
all_labels = []

# ========= 依次读取 9 个 ORC =========
for fits_file, cat_file in zip(fits_paths, catalogue_paths):
    print(f"Processing {fits_file} ...")

    ds = CataData(
        catalogue_paths=[cat_file],
        image_paths=[fits_file],
        field_names=["ORCfield"],
        cutout_shape=70,   # centre-crop 70x70
        transform=transform,
        targets=["ORC"],
    )

    # 每个文件只有 1 个 ORC
    img, label = ds[0]    # img shape: (1,70,70) 或 (70,70)
    print(img.shape)

    all_images.append(img)
    all_labels.append(label)

# ========= 合并成一个 tensor =========
images_tensor = torch.stack(all_images)   # shape = (9,1,70,70)

# # ========= 保存到 pt 文件 =========
save_dict = {
    "images": images_tensor,
    "labels": all_labels,
}

torch.save(save_dict, "orc_cutouts.pt")

print("Saved ORC cutouts to orc_cutouts.pt")
print("Shape:", images_tensor.shape)