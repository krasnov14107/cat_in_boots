
# coding: utf-8

# In[79]:


# импортируем необходимые библиотеки для обработки штрих-кодов
import numpy as np
import argparse
import cv2
 
# создаём парсер аргументов командной строки и парсим их
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "path to the image file")
#args = vars(ap.parse_args())


# In[80]:


#загружаем изображение и конвертируем его в градации серого
image = cv2.imread('beard_trimmer.jpg')#(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

final_wide = 800
r = float(final_wide) / image.shape[1]
dim = (final_wide, int(image.shape[0] * r))

resize = cv2.resize(gradient, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resize)
cv2.waitKey(0)


# In[81]:


# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))#сглаживание высокочастотного
#шума на нашей картинке
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
#бинаризация изображения

resiz = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resiz)
cv2.waitKey(0)


# In[82]:


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

resi = cv2.resize(closed, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize", resi)
cv2.waitKey(0)


# In[83]:


# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

res = cv2.resize(closed, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize im", res)
cv2.waitKey(0)


# In[84]:


# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
# compute the rotated bounding box of the largest contour

rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

#peri = cv2.arcLength(c, True)
#approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
#cv2.imshow("Image", image)
#cv2.waitKey(0)


# In[85]:


print(image.shape)
 
result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resize image", result)
cv2.waitKey(0)

