
# coding: utf-8

# In[81]:


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


# In[82]:


# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#cv2.imshow("Gray image", gray)
#cv2.waitKey(0)& 0xFF


# In[83]:


# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))#сглаживание высокочастотного
#шума на нашей картинке
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


# In[84]:


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# In[85]:


# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


# In[86]:


#cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
#c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
#rect = cv2.minAreaRect(c)
#box = np.int0(cv2.boxPoints(rect))

#cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

#cv2.imshow("Image", image)
#cv2.waitKey(0)& 0xFF


# In[87]:


h = 640/10;
ar = [];
for i in range(10):
    ar.append(0+i*h)
ar.append(640)
s = 662/10;
ar2 = [];
for j in range(10):
    ar2.append(0+j*s)
ar2.append(662)
# Draw a black lines with thickness of 2 px
#img = cv2.line(image,(0,0),(640,0),(0,0,0),2)
#img2 = cv2.line(image,(0,0),(0,662),(0,0,0),2)
#img3 = cv2.line(image,(0,662),(640,662),(0,0,0),2)
#img4 = cv2.line(image,(640,0),(640,662),(0,0,0),2)
#cv2.imshow("Image", image)
#cv2.waitKey(0)& 0xFF
ar


# In[88]:


ar2


# In[89]:


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
x1 = [];
y1 = [];
x2 = [];
y2 = [];
for edge2 in ar[0:-1]:
    for edge1 in ar2[0:-1]:
        crop_img = gray[int(edge2):int(edge2+s),int(edge1):int(edge1+h)];
        edges = cv2.Canny(crop_img,10,10,apertureSize = 3);
        minLineLength = 15
        maxLineGap = 2
        lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap);
        if lines is not None: 
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(image,(int(edge2)+x1,int(edge1)+y1),(int(edge2)+x2,int(edge1)+y2),(0,255,0),2)
x1 = [];
y1 = [];
x2 = [];
y2 = [];
cv2.imshow("Image", image)
cv2.waitKey(0)& 0xFF

