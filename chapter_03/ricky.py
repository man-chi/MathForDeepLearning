import numpy as np
import matplotlib.pylab as plt
from PIL import Image

# Create a synthetic face-like image with varying gray levels
np.random.seed(42)
face_img = np.zeros((512, 512), dtype=np.uint8)
# Create gradient and noise pattern to simulate a complex image
for i in range(512):
    for j in range(512):
        face_img[i, j] = int(128 + 50 * np.sin(i / 50) * np.cos(j / 50) + np.random.normal(0, 20))
face_img = np.clip(face_img, 0, 255).astype(np.uint8)

im = face_img[:512, :512]
Image.fromarray(im).save("ricky.png")
hr, xr = np.histogram(im, bins=256)
hr = hr / hr.sum()

# Create a synthetic ascent-like image (diagonal gradient)
ascent_img = np.zeros((512, 512), dtype=float)
for i in range(512):
    for j in range(512):
        ascent_img[i, j] = (i + j) / 4 + np.random.normal(0, 10)
ascent_img = np.clip(ascent_img, 0, 255).astype(np.uint8)

im = ascent_img
Image.fromarray(im).save("ascent.png")
ha, xa = np.histogram(im, bins=256)
ha = ha / ha.sum()

plt.plot(xr[:-1], hr, color='k', label="Face")
plt.plot(xa[:-1], ha, linestyle=(0, (1, 1)), color='k', label="Ascent")
plt.legend(loc="upper right")
plt.xlabel("Gray level")
plt.ylabel("Probability")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("ricky_probability.png", dpi=300)
plt.show()
plt.close()
