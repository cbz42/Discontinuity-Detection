import cv2
import numpy as np
from skimage.exposure import rescale_intensity

image = cv2.imread('sdf.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(iH, iW) = image.shape[:2]

Mask = np.array((
	[-1, -1, -1],
	[-1, 8, -1],
	[-1, -1, -1]), dtype="int")

pad = 1
output = np.zeros((iH, iW), dtype="float32")
gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)

for y in np.arange(pad,iH+pad):
	for x in np.arange(pad,iW+pad):
		roi = gray[y - pad:y + pad + 1, x - pad:x + pad + 1]
		#print(roi.shape)
		k = (roi * Mask).sum()
		if k == 1:
			output[y - pad, x - pad] = k
		else:
			output[y - pad, x - pad] = 0

sdf = rescale_intensity(output, in_range=(0, 255))
sdf1 = (sdf * 255).astype("uint8")
#print(sdf1)
cv2.imshow("hello",sdf1)
cv2.waitKey(0)

