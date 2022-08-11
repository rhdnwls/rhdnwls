#from cProfile import label
from ctypes.wintypes import RGB
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
'''
#1,다양한 방법의 그래프 만들기
x = np.array([10, 20, 15])
y = np.array([5, 10, 15])

plt.plot(x, y, 'b-o', label='Data 1')
plt.ylabel('Y axls')
plt.xlabel('X axls')
plt.legend()
plt.show()
'''
'''
#2, 라인 그래프 만들기
x = np.linspace(0, 2, 100)# 0부터 2까지는 100개로 나누기
y1 = 0.5 * x 
y2 = 0.5 * x**2# 기울기가 0.5인 2차 함수 
y3 = 0.5 * x**3

plt.plot(x, y1, label='linear')
plt.plot(x, y2, label='quadratic')
plt.plot(x, y3, label='cubic')

plt.legend()
plt.show()
'''

#3 BGR & RGB
#1, 이미지 불러오기
image = cv.imread("Colorful.png")

#BGR
plt.figure()
plt.imshow(image)
plt.title("Original")

#rgb
rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.figure()
plt.imshow(rgb)
plt.title("RGB")
# plt.show()


#5 Convert to the gray

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray, cmap = 'gray')
plt.title("GRAY")
#plt.show()

#blur
blur = cv.blur(image, (50,50))
blur = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(rgb)
plt.title("RGB")
plt.subplot(122)
plt.imshow(blur)
plt.title("BLUR")
#plt.show()

#7 Edge Detection
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray, cmap = 'gray')
plt.title("Gray")

edges = cv.Canny(gray, 100, 200)
plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title("Gray")
plt.subplot(122)
plt.imshow(edges, cmap = 'gray')
plt.title("Edge Detection")
plt.show()