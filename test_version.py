import cv2

img = cv2.imread('1.jpg')
# cv2.imread() 用來讀取圖片檔案，這裡讀取的是 1.jpg。
# 回傳值是一個 NumPy 陣列 (ndarray)，格式是 BGR（不是 RGB）。
# shape 通常是 (height, width, 3)，最後的 3 代表三個色彩通道 (Blue, Green, Red)。
img = cv2.resize(img,(200,200))
# cv2.imshow 只能在 桌面環境（有 GUI 的作業系統，例如 Windows, macOS, Ubuntu GUI）下使用。
# 在 Jupyter Notebook 或某些遠端環境（如 Colab）會出錯，因為沒有 GUI 視窗。

cv2.imshow('視窗名稱',img)

cv2.waitKey(0)
# cv2.waitKey(ms) 會等待鍵盤輸入，參數是等待的毫秒數。
# 0 表示「無限等待」，直到使用者按下任意鍵才繼續。
# 如果你寫 cv2.waitKey(1000) 就會等 1 秒。

cv2.destroyAllWindows()
# 關閉所有由 OpenCV 建立的視窗。


