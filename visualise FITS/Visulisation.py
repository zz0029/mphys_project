import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# Input parameters
IMG_PATH = "../ORC FITS image/image.i.EMU_2034-60.SB72006.cont.taylor.0.restored.conv.fits"
ORC_RA_DEG  = 315.74292
ORC_DEC_DEG = -62.00444

# Read the ORC FITS image and get HDUList
hdul = fits.open(IMG_PATH)
print(hdul.info())
print("\n")

# Get the header and data
hdu_number = 0
hdu = hdul[hdu_number]
hdu_data = hdul[hdu_number].data.squeeze()
hdu_header = hdul[hdu_number].header

print(hdu_header.tostring(sep='\n')) #print header
# print(hdu_data) #print data



# Built WCS object
wcs = WCS(hdu_header).celestial
c = wcs.pixel_to_world(0, 0)
print(c.ra.deg, c.dec.deg)

# Image Normalisation
mean, med, std = sigma_clipped_stats(hdu_data, sigma=3.0)
norm = ImageNormalize(hdu_data,
                      interval=PercentileInterval(99.5),   # 丢掉极端0.5%
                      stretch=AsinhStretch())
#Plot image
fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
im=ax.imshow(hdu_data, origin='lower', norm=norm, cmap='gray')

ax.grid(color='white', ls='solid')
ax.set(xlabel='RA', ylabel='Dec')
fig.colorbar(im, ax=ax, label=hdu_header.get('BUNIT',''))

# Zoom at ORC
# 1) ORC中心（世界坐标）
ORC = SkyCoord(ORC_RA_DEG*u.deg, ORC_DEC_DEG*u.deg, frame='fk5')

# 2) 世界 -> 像素
cx, cy = wcs.world_to_pixel(ORC)   # wcs = WCS(hdu_header).celestial（2D）

# 3) 像素标尺（arcsec/像素）
from astropy.wcs.utils import proj_plane_pixel_scales
scales_deg = proj_plane_pixel_scales(wcs)   # deg/pix, order ~ (y, x)
sx_as = scales_deg[1]*3600.0
sy_as = scales_deg[0]*3600.0

# 4) 3 arcmin 半径 → 像素
r_as = 3.0*60.0
rx = r_as / sx_as
ry = r_as / sy_as

# 5) 设定视野窗口（以 ORC 为中心、3' 半径）
ax.set_xlim(cx - rx, cx + rx)
ax.set_ylim(cy - ry, cy + ry)

# 6) 可选：画 3' 圆
from astropy.visualization.wcsaxes.patches import SphericalCircle
ax.add_patch(SphericalCircle(ORC, 3*u.arcmin, edgecolor='w', facecolor='none', lw=1))

# Plot cotour
levels = med + std * np.array([3,5,8,12])   # 用你前面算的 med/std
ax.contour(hdu_data, levels=levels, colors='red', linewidths=0.3, alpha=0.8)

# Label the coordinates
# ax.scatter(ORC_RA_DEG, ORC_DEC_DEG,
#            s=80, facecolors='none', edgecolors='blue', linewidths=1.8,
#            transform=ax.get_transform('world'), zorder=5)

plt.show()