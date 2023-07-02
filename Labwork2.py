#!/usr/bin/env python
# coding: utf-8

# In[51]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

#Task 1:
#read image
img = cv2.imread(r"C:\Users\thanh\Desktop\DIP\grey.jpg")
#plt.imshow(img)



# In[52]:


#syntax: image, channels (for grayimg = 0), mask (find hist for full img = None), histSize, ranges
hist = cv2.calcHist(img, [0], None, [256], [0,256])

#Show
plt.plot(hist)


# In[53]:


#Task 2:

#Apply average filter
k_ave=(10,10) #kernel size
img_ave = cv2.blur(img, k_ave)

#Apply Gaussian filter
k_Gau=(3,3) #kernel size
img_Gau = cv2.GaussianBlur(img, k_Gau,0)

#Apply median filter
img_med = cv2.medianBlur(img, 3)

#Show results and compare
fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(img)
axs[0, 0].set_title("Original")
axs[0, 0].axis("off")

axs[1, 0].imshow(img_ave)
axs[1, 0].set_title("Average Filter")
axs[1, 0].axis("off")

axs[0, 1].imshow(img_Gau)
axs[0, 1].set_title("Gaussian filter")
axs[0, 1].axis("off")

axs[1, 1].imshow(img_med)
axs[1, 1].set_title("Median filter")
axs[1, 1].axis("off")

# In[54]:


#Task 3:
#Create Gaussian noise
# Calculate the mean and standard deviation of pixel values in the image
mean = np.mean(img)
std_dev = np.std(img)
#Add noise
gaussian_noise = np.random.normal(mean, std_dev, img.shape)

#Add Gaussian noise
img_Gau_noise = img + gaussian_noise

# Normalize the pixel values to the range [0, 1]
img_Gau_noise = img_Gau_noise / 255

#Add Salt and Pepper noise
#probability of noise
prob = 0.01
#Create a matrix of zeros with the same shape as the image
img_snp_noise = np.zeros(img.shape, np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rnd = np.random.random()
        #If the random number is less than the noise probability, add salt and pepper noise
        if rnd < prob / 2:
            img_snp_noise[i][j] = 0
        elif rnd < prob:
            img_snp_noise[i][j] = 255
        else:
            img_snp_noise[i][j] = img[i,j]

# Normalize the pixel values to the range [0, 1]
img_snp_noise = img_snp_noise / 255.0

#Add Periodic noise
# Generate the periodic noise
per = np.random.normal(mean, std_dev, img.shape)

# Resize the noise to match the shape of the image
per = cv2.resize(per, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Scale the noise to have values between 0 and 1
per = (per - np.min(per)) / (np.max(per) - np.min(per))

# Convert the image to float32 for compatibility with addWeighted function
img = img.astype(np.float32)

# Convert the noise to float32 for compatibility with addWeighted function
per = per.astype(np.float32)

# Add the noise to the image
img_per_noise = cv2.addWeighted(img, 1, per, 0.5, 0)

# Convert the result back to uint8 for display
img_per_noise = img_per_noise.astype(np.uint8)

# Normalize the pixel values to the range [0, 1]
img_per_noise = img_per_noise /255.0

#Calculate histogram of each
hist_Gau_noise = cv2.calcHist(img_Gau_noise.astype(np.uint8), [0], None, [256], [0,256])
hist_snp_noise = cv2.calcHist(img_snp_noise.astype(np.uint8), [0], None, [256], [0,256])
hist_per_noise = cv2.calcHist(img_per_noise.astype(np.uint8), [0], None, [256], [0,256])

#Show results and compare
fig, axs = plt.subplots(3, 2)

axs[0, 0].imshow(img_Gau_noise)
axs[0, 0].set_title("Gaussian noise")
axs[0, 0].axis("off")

axs[1, 0].imshow(img_snp_noise)
axs[1, 0].set_title("Salt and Pepper noise")
axs[1, 0].axis("off")

axs[2, 0].imshow(img_per_noise)
axs[2, 0].set_title("Prediodic noise")
axs[2, 0].axis("off")

axs[0, 1].plot(hist_Gau_noise)
axs[0, 1].set_title("Gaussian noise histogram")
axs[0, 1].axis("off")

axs[1, 1].plot(hist_snp_noise)
axs[1, 1].set_title("Salt and Pepper noise histogram")
axs[1, 1].axis("off")

axs[2, 1].plot(hist_per_noise)
axs[2, 1].set_title("Periodic noise histogram")
axs[2, 1].axis("off")


# In[55]:


#Task 4

#Apply average filter
k_ave=(10,10) #kernel size
img_per_denoise = cv2.blur(img_snp_noise, k_ave)

#Apply Gaussian filter
k_Gau=(3,3) #kernel size
img_Gau_denoise = cv2.GaussianBlur(img_snp_noise, k_Gau,0)

#Apply median filter
img_snp_denoise = cv2.medianBlur(img_snp_noise.astype(np.float32), 3)

#Show results and compare
fig, axs = plt.subplots(3, 2)

axs[0, 0].imshow(img_Gau_noise)
axs[0, 0].set_title("Gaussian noise")
axs[0, 0].axis("off")

axs[1, 0].imshow(img_snp_noise)
axs[1, 0].set_title("Salt and Pepper noise")
axs[1, 0].axis("off")

axs[2, 0].imshow(img_per_noise)
axs[2, 0].set_title("Prediodic noise")
axs[2, 0].axis("off")

axs[0, 1].imshow(img_Gau_denoise)
axs[0, 1].set_title("Gaussian denoise")
axs[0, 1].axis("off")

axs[1, 1].imshow(img_snp_denoise)
axs[1, 1].set_title("Salt and Pepper denoise")
axs[1, 1].axis("off")

axs[2, 1].imshow(img_per_denoise)
axs[2, 1].set_title("Periodic denoise")
axs[2, 1].axis("off")



# In[56]:


#Task 5:
img_per_gray = cv2.cvtColor(img_per_noise.astype(np.uint8), cv2.COLOR_BGR2GRAY)

#Perform Fourier Transform on the image
img_per_fft = np.fft.fft2(img_per_noise)
img_per_shift = np.fft.fftshift(img_per_fft)
magnitude_spectrum = 20*np.log(np.abs(img_per_shift))

#Create a high-pass filter to remove the noise:
rows, cols, ch = img_per_shift.shape
crow, ccol = rows//2, cols//2
r = 80
mask_gray = np.ones((rows, cols), np.uint8)
x, y = np.ogrid[:rows, :cols]
mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
mask_gray[mask_area] = 0
mask = np.broadcast_to(mask_gray[..., np.newaxis], img_per_shift.shape)

#Apply the high-pass filter to the Fourier Transform of the image:
img_per_filtered = img_per_shift * mask
magnitude_spectrum_filtered = 20*np.log(np.abs(img_per_filtered))

#Perform Inverse Fourier Transform on the filtered image:
img_per_ishift = np.fft.ifftshift(img_per_filtered)
img_per_denoise_2 = np.fft.ifft2(img_per_ishift)
img_per_denoise_2 = np.abs(img_per_denoise_2)

#Show results and compare
fig, axs = plt.subplots(1,2)

axs[0].imshow(img_per_noise)
axs[0].set_title("Periodic noise")
axs[0].axis("off")

axs[1].imshow(img_per_denoise_2)
axs[1].set_title("Periodic denoise")
axs[1].axis("off")

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
