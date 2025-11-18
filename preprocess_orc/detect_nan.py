import torch
d = torch.load("orc_cutouts.pt")
imgs = d["images"].float()
print(torch.isnan(imgs).sum())