import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2

db = h5py.File('SynthText.h5', 'r')
im_names = list(db['data'].keys())

im = im_names[0]
img = db['data'][im][:]

height = img.shape[0]
width = img.shape[1]

font = db['data'][im].attrs['font']
txt = db['data'][im].attrs['txt']
charBB = db['data'][im].attrs['charBB']
wordBB = db['data'][im].attrs['wordBB']
print(wordBB)

cv2.line(img, (int(charBB[0][0][0]),int(charBB[1][0][0])), (int(charBB[0][1][0]),int(charBB[1][1][0])), (255,0,0), 1)
cv2.line(img, (int(charBB[0][1][0]),int(charBB[1][1][0])), (int(charBB[0][2][0]),int(charBB[1][2][0])), (255,0,0), 1)
cv2.line(img, (int(charBB[0][2][0]),int(charBB[1][2][0])), (int(charBB[0][3][0]),int(charBB[1][3][0])), (255,0,0), 1)
cv2.line(img, (int(charBB[0][3][0]),int(charBB[1][3][0])), (int(charBB[0][0][0]),int(charBB[1][0][0])), (255,0,0), 1)

# cv2.line(img, (int(wordBB[1][0][0]),int(wordBB[1][0][1])), (int(wordBB[1][1][0]),int(wordBB[1][1][1])), (0,255,0), 1)
# cv2.line(img, (int(wordBB[1][1][0]),int(wordBB[1][1][1])), (int(wordBB[1][2][0]),int(wordBB[1][2][1])), (0,255,0), 1)
# cv2.line(img, (int(wordBB[1][2][0]),int(wordBB[1][2][1])), (int(wordBB[1][3][0]),int(wordBB[1][3][1])), (0,255,0), 1)
# cv2.line(img, (int(wordBB[1][3][0]),int(wordBB[1][3][1])), (int(wordBB[1][0][0]),int(wordBB[1][0][1])), (0,255,0), 1)
cv2.line(img, (int(wordBB[1][0][0]),int(wordBB[1][1][0])), (int(wordBB[1][2][0]),int(wordBB[1][2][1])), (0,255,0), 1)

plt.imshow(img)
plt.title('Image')
plt.show()


