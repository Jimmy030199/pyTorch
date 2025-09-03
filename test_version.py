import cv2

img = cv2.imread('1.jpg')
img = cv2.resize(img,(200,200))

cv2.imshow('視窗名稱',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

