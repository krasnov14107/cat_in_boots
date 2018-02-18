
# coding: utf-8

# In[5]:


# -*- coding: utf-8 -*-
# импортируйте необходимые пакеты
import numpy as np
import cv2

# загрузите изображение, смените цвет на оттенки серого и уменьшите резкость
image = cv2.imread("books.jpg")
#cv2.imshow("Original image",image)
#cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imwrite("gray.jpg", gray)
#cv2.imshow("Gray image", gray)
#cv2.waitKey(0)


# In[4]:


# распознавание контуров
edged = cv2.Canny(gray, 10, 250)
cv2.imwrite("edged.jpg", edged)
#cv2.imshow("Edged image", edged)
#cv2.waitKey(0)


# In[6]:


# создайте и примените закрытие
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed.jpg", closed)


# In[7]:


# найдите контуры в изображении и подсчитайте количество книг
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
total = 0


# In[8]:


# цикл по контурам
for c in cnts:
    # аппроксимируем (сглаживаем) контур
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # если у контура 4 вершины, предполагаем, что это книга
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1


# In[15]:


# показываем результирующее изображение
print("Я нашёл {0} книг на этой картинке".format(total)) 
cv2.imwrite("output.jpg", image)
cv2.imshow("Result image", image)
cv2.waitKey(0)

