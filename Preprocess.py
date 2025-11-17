from cata2data import CataData
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

ORC_data = CataData(
    catalogue_paths=["ORC coordinates.txt"],
    image_paths=["fits image/image.i.EMU_2034-60.SB72006.cont.taylor.0.restored.conv.fits"],
    field_names=["MYFIELD"],                 # give a name of field
    cutout_shape=70,                         # wanted pixel sizes
    targets = ['ORC']
)


# data of cutout
print(ORC_data[0][0].shape) #shape
print(ORC_data[0][1]) #label

# cutout image
idx = 0
ORC_data.plot(idx)
ORC_data.df.iloc[idx: idx + 1]
