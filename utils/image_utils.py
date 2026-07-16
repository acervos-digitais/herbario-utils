import matplotlib.cm as cm

import numpy as np

from PIL import Image as PImage

def scale_2d_array(array, size, sampling=PImage.Resampling.BILINEAR):
  if len(array.shape) == 1:
    dim = int(array.shape[0] ** 0.5)
    array = array.reshape(dim, dim)
  return np.array(PImage.fromarray(array).resize(size, resample=sampling))

def mask_image(img, mask, sampling=PImage.Resampling.BILINEAR):
  img_np = np.array(img)

  if len(mask.shape) < 2 or mask.shape[0] != img_np.shape[0] or mask.shape[1] != img_np.shape[1]:
    mask = scale_2d_array(mask, img.size, sampling=sampling)

  if len(mask.shape) == 2:
    mask = mask[:, :, None]

  return PImage.fromarray((mask * img_np).astype(np.uint8))

# map := [ 'viridis', 'plasma', 'inferno', 'magma' ]
def heatmap_image(data, *, size=None, cmap="inferno", sampling=PImage.Resampling.BILINEAR):
  if size:
    data = scale_2d_array(data, size, sampling=sampling)
  map_fun_np = np.vectorize(cm.get_cmap(cmap))
  rgba_np = 255 * np.stack(map_fun_np(data)[:3], axis=-1)
  return PImage.fromarray(rgba_np.astype(np.uint8)).resize(size)
