#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


#read image
img = cv2.imread(r"C:\Users\thanh\Desktop\DIP\grey.jpg")


# In[2]:


#Task 1:
#Apply Laplacian filter
img_lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3, scale=1)
# Convert the result to uint8 format before displaying
img_lap = np.uint8(np.absolute(img_lap))

# Apply Sobel filter along x and y directions with a kernel size of 3
img_solx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
img_soly = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

# Convert the results to uint8 format before displaying
img_solx = np.uint8(np.absolute(img_solx))
img_soly = np.uint8(np.absolute(img_soly))

# Combine the filtered images using bitwise OR operation
img_sol = cv2.bitwise_or(img_solx, img_soly)


#plt.imshow(img)
#plt.title("Original")
#plt.axis("off")
cv2.namedWindow("Win1")
cv2.setWindowTitle("Win1", "Original")
cv2.imshow("Win1", img)

#Show results and compare
fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(img_lap)
axs[0, 0].set_title("Laplacian")
axs[0, 0].axis("off")

axs[1, 0].imshow(img_solx)
axs[1, 0].set_title("Sobel x")
axs[1, 0].axis("off")

axs[0, 1].imshow(img_sol)
axs[0, 1].set_title("Sobel")
axs[0, 1].axis("off")

axs[1, 1].imshow(img_soly)
axs[1, 1].set_title("Sobel y")
axs[1, 1].axis("off")

# In[3]:

#Task 2:
#Apply Canny
img_can = cv2.Canny(img, 100, 200)

#Show results and compare:
fig, axs = plt.subplots(1,3)

axs[0].imshow(img_lap)
axs[0].set_title("Laplacian")
axs[0].axis("off")

axs[1].imshow(img_sol)
axs[1].set_title("Sobel")
axs[1].axis("off")

axs[2].imshow(img_can,cmap="gray")
axs[2].set_title("Canny")
axs[2].axis("off")


# In[7]:


#Task 3:
# Detect lines using the Hough transform
lines = cv2.HoughLines(img_can, 1, np.pi/180, 50)

# Draw detected lines on the original image
image_Hou = img.copy()
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #Apply cv2.line()
    cv2.line(image_Hou, (x1,y1), (x2,y2), (0,0,255), 2)
    
#Show results and compare:
fig, axs = plt.subplots(1,2)

axs[0].imshow(img)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(image_Hou)
axs[1].set_title("Hough transform")
axs[1].axis("off")

# In[8]:


#Task 4:
# Create a mask with the same size as the input image and set all values to zero
mask = np.zeros(img.shape[:2],np.uint8)

# Define the rectangle enclosing the object of interest
rect = (50,50,450,290)

# Apply GrabCut algorithm to segment the image based on the rectangle and mask
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# Create a binary mask where 0s represent background and 1s represent object of interest
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Apply the binary mask to the original image to obtain the segmented image
img_grab = img*mask2[:,:,np.newaxis]

# Apply global thresholding
thresh_value, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#Show results and compare:
fig, axs = plt.subplots(1,2)

axs[0].imshow(img_grab)
axs[0].set_title("Image Segmentation")
axs[0].axis("off")

axs[1].imshow(img_thresh)
axs[1].set_title("Global Thresholding")
axs[1].axis("off")

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
