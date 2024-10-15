import numpy as np
import cv2

# Load the image
img = cv2.imread('images/0h5ya8-0.png')

# Define the rectangle for the object of interest
rect = (50,50,450,290)

# Create a mask with the rectangle
mask = np.zeros(img.shape[:2],np.uint8)

# Apply the GrabCut algorithm
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# Create a mask for the foreground
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Apply the mask to the original image
img = img*mask2[:,:,np.newaxis]

# Display the result
cv2.imshow("Segmented Image", img)
cv2.waitKey(0)