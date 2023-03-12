prefix = 'cabin4'

import numpy as np
import cv2

msk = cv2.imread(f"{prefix}_mask.png", 0) / 255
depth = np.load(f"{prefix}.npy")
im = cv2.imread(f"{prefix}.png")
depth = cv2.resize(depth, (512, 512))
print(msk.shape, depth.shape, im.shape)

positions = np.nonzero(msk)

top = positions[0].min()
bottom = positions[0].max()
left = positions[1].min()
right = positions[1].max()

new_mask = np.zeros((512, 512))
new_depth = np.zeros((512, 512))
new_im = np.zeros((512, 512, 3))
new_l = 256 - (right - left) // 2
new_r = new_l + right - left
new_top = 256 - (bottom - top) // 2
new_bottom = new_top + bottom - top

new_mask[new_top:new_bottom, new_l:new_r] = msk[top:bottom, left:right]
new_depth[new_top:new_bottom, new_l:new_r] = depth[top:bottom, left:right]
new_im[new_top:new_bottom, new_l:new_r, :] = im[top:bottom, left:right, :]
cv2.imwrite(f"{prefix}_centered_mask.png", new_mask * 255)
np.save(f"{prefix}_centered.npy", new_depth)
cv2.imwrite(f"{prefix}_centered.png", new_im)
