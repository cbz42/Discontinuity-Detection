import cv2
import numpy as np
from skimage.exposure import rescale_intensity

image = cv2.imread('Rest.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(iH, iW) = image.shape[:2]

Mask1= np.array((
	[-1,-1,-1],
	[2,2,2],
	[-1, -1, -1]), dtype="int")
Mask2 = np.array((
	[2,-1,-1],
	[-1,2,-1],
	[-1,-1,2]), dtype="int")

Mask3 = np.array((
	[-1,2,-1],
	[-1,2,-1],
	[-1,2,-1]), dtype="int")

Mask4 = np.array((
	[-1,-1,2],
	[-1,2,-1],
	[2,-1,-1]), dtype="int")


pad = 1
output = np.zeros((iH, iW), dtype="float32")
gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)

for y in np.arange(pad,iH+pad):
	for x in np.arange(pad,iW+pad):
		roi = gray[y - pad:y + pad + 1, x - pad:x + pad + 1]
		#print(roi.shape)
		k1 = (roi * Mask1).sum()
		k2 = (roi * Mask2).sum()
		k3 = (roi * Mask3).sum()
		k4 = (roi * Mask4).sum()
		k = max(k1,k2,k3,k4)
		output[y - pad, x - pad] = k

sdf = rescale_intensity(output, in_range=(0, 255))
sdf1 = (sdf * 255).astype("uint8")
#print(sdf1)
cv2.imshow("hello",sdf1)
cv2.waitKey(0)

