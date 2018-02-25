
# coding: utf-8

# In[30]:


# добавим необходимый пакет с opencv
from matplotlib import pyplot as plt
import numpy as np
import cv2
 
# загружаем изображение и отображаем его
image = cv2.imread('astr_1_14.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
#cv2.imwrite("gray.jpg", gray)

#plt.imshow(image, cmap='gray')
#plt.show()

cv2.imshow("Original image", image)
cv2.waitKey(0)
print(image.shape)


# In[31]:


# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("Gray image", gray)
cv2.waitKey(0)


# In[32]:


# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))#сглаживание высокочастотного
#шума на нашей картинке
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


# In[33]:


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# In[34]:


# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


# In[35]:


cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

cv2.imshow("Image", image)
cv2.waitKey(0)

