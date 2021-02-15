import cv2
import numpy as np

img = cv2.imread("pan_card.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (5, 5))

high, th_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

canny = cv2.Canny(th_image, 0.5 * high, high)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
'''

idx = 0
images = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 50:
        idx += 1
        new_img = img[y:y + h, x:x + w]
        images.append(new_img)

for i in images:
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(i, -1, sharpen_kernel)
    cv2.imwrite('processed image'+'.jpeg',im)

'''
pts = np.argwhere(canny > 0)

y1, x1 = pts.min(axis=0)
y2, x2 = pts.max(axis=0)

cropped = img[y1:y2, x1:x2]
tag = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3,cv2.LINE_AA)

cv2.imshow('cropped',tag)

'''
