
# coding: utf-8

# In[82]:


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

plt.rcParams["figure.figsize"] = (66,64)

ed = cv2.Canny(image,100,450,apertureSize = 3)
plt.subplot(122),plt.imshow(ed,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# In[83]:


# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#cv2.imshow("Gray image", gray)
#cv2.waitKey(0)& 0xFF


# In[84]:


# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))#сглаживание высокочастотного
#шума на нашей картинке
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


# In[85]:


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


# In[86]:


# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


# In[87]:


#cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
#c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
 
#rect = cv2.minAreaRect(c)
#box = np.int0(cv2.boxPoints(rect))

#cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

#cv2.imshow("Image", image)
#cv2.waitKey(0)& 0xFF


# In[88]:


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


# In[89]:


ar2


# In[92]:


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
x1 = [];
y1 = [];
x2 = [];
y2 = [];
ar3 = [];
ar3.append(0.0);
ar3.append(662.0);
q = ar2[10];
#edge1-горизонтальная координата в ar[]
#edge2-вертикальная координата в ar2[]
for edge1 in ar[0:-1]:
    for edge2 in ar3[0:-1]:
        crop_img = gray[int(ar3[0]):int(ar3[1]),int(edge1):int(edge1+h)];
        edges = cv2.Canny(crop_img,100,450,apertureSize = 3);
        
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Slice Image'), plt.xticks([]), plt.yticks([]) 
        plt.show() 
         
#         lines = cv2.HoughLines(edges,1,np.pi/180,60)
#         if lines is not None:
#             for line in lines:
#                 for rho,theta in lines:
#                     a = np.cos(theta)
#                     b = np.sin(theta)
#                     x0 = a*rho
#                     y0 = b*rho
#                     x1 = int(x0 + 10*(-b))
#                     y1 = int(y0 + 10*(a))
#                     x2 = int(x0 - 10*(-b))
#                     y2 = int(y0 - 10*(a))
#                     cv2.line(image,(int(edge2)+x1,int(edge1)+y1),(int(edge2)+x2,int(edge1)+y2),(0,255,0),2)
        
        minLineLength = 30
        maxLineGap = 5
        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap);
        if lines is not None: 
            for line in lines:
                for x1,y1,x2,y2 in line:
                    ctr_points = cv2.line(gray,(int(edge1)+x1,int(edge2)+y1),(int(edge1)+x2,int(edge2)+y2),(0,255,0),2)
cv2.imshow("Image", gray)
cv2.waitKey(0)& 0xFF


# In[67]:


line


# In[68]:


lines


# In[69]:


edges


# In[79]:


ctr_points

