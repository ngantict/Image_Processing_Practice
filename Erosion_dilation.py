import cv2
import numpy as np
from matplotlib import pyplot as plt

I = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

S = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]], dtype=np.uint8)

img_erosion = cv2.erode(I, S, iterations=1)
img_dilation = cv2.dilate(I, S, iterations=1)

img_opening = cv2.dilate(img_erosion, S, iterations=1)
img_closing = cv2.erode(img_dilation, S, iterations=1)

#cv2.imshow('Original image', I)
#cv2.imshow('Structure element', S)
#cv2.imshow('Erosion', img_erosion)
#cv2.imshow('Dilation', img_dilation)
#cv2.imshow('Opening', img_opening)
#cv2.imshow('Closing', img_closing)

fig, axs = plt.subplots(3, 2)

axs[0, 0].imshow(I)
axs[0, 0].set_title("Original")
axs[0, 0].axis("off")

axs[0, 1].imshow(S)
axs[0, 1].set_title("Structure element")
axs[0, 1].axis("off")

axs[1, 0].imshow(img_erosion)
axs[1, 0].set_title("Erosion")
axs[1, 0].axis("off")

axs[1, 1].imshow(img_dilation)
axs[1, 1].set_title("Dilation")
axs[1, 1].axis("off")

axs[2, 0].imshow(img_opening)
axs[2, 0].set_title("Opening")
axs[2, 0].axis("off")

axs[2, 1].imshow(img_closing)
axs[2, 1].set_title("Closing")
axs[2, 1].axis("off")

plt.show()
  
cv2.waitKey(0)
