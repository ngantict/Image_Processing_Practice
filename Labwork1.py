import cv2
import numpy as np
from matplotlib import pyplot as plt

#Task 1:
#read image
img = cv2.imread(r"C:\Users\thanh\Desktop\DIP\grey.jpg")
#plt.imshow(img)
cv2.namedWindow("Win1")
cv2.imshow("Win1", img)

#Task 2:
#scale image
scale = 60 #scale percent 60%
width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)
#dimention
dim = (width, height)

#resize image
img_resize = cv2.resize(img,dim)

#show result
#plt.imshow(img_resize)
cv2.namedWindow("Win2")
cv2.imshow("Win2", img_resize)

#Task 3
# Define the constants
a = 1.5 # increase brightness by 50%
b = 10 # add a constant value to each pixel

# Apply the formula, clipping the results to the valid range of 0-255
img_bright = np.clip(a * img + b, 0, 255).astype(np.uint8)

# Display the processed image
#plt.imshow(img_bright)
cv2.namedWindow("Win2")
cv2.imshow("Win2", img_bright)

#Task 4:
#syntax: image, channels (for grayimg = 0), mask (find hist for full img = None), histSize, ranges
hist = cv2.calcHist(img, [0], None, [256], [0,256])

#Show
plt.plot(hist)
# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#equalization
equ = cv2.equalizeHist(img_gray)

#Show result
#plt.imshow(equ, cmap='gray')
#plt.axis('off')
cv2.namedWindow("Win3")
cv2.imshow("Win3", equ)

#Task 5:
# Apply global thresholding
thresh_value, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Show result
#plt.imshow(img_thresh)
cv2.namedWindow("Win4")
cv2.imshow("Win4", img_thresh)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
