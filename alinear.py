# MOD: Mauri550
# Sources:
# - Correct skew: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# - FloodFill borders: https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# - Autocrop:
#   - https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
#   - https://stackoverflow.com/questions/37803903/opencv-and-python-for-auto-cropping


# Import the necessary packages
import numpy as np
import argparse
import cv2
 
# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
ap.add_argument("-o", "--out", required=True,
	help="path to output image file")
ap.add_argument("-n", "--negativeout", required=True,
	help="path to output negative image file")
args = vars(ap.parse_args())
 
imageNameOut = args["out"]
imageNameNegativeOut = args["negativeout"]

# Load the image from disk
imageName = args["image"]
imageBig = cv2.imread(imageName)

#Resize Original Image to 60% 
image = cv2.resize(imageBig, (0,0), fx=0.6, fy=0.6) 

# Convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# Threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# FloodFill external borders
cv2.floodFill(thresh, mask, (0,0), 0)

# Grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
 
# The `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# Otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# Rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Show the angle correction
print("[INFO] angle: {:.3f}".format(angle))

# Convert the image to grayscale (again) and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# FloodFill external borders
cv2.floodFill(thresh, mask, (0,0), 0)

# Find contours
im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("[INFO] Contours {len}".format(len = len(contours)))

# Get best contours
maxH, maxW = rotated.shape[:2]
best_box=[-1,-1,-1,-1]
for c in contours:
   x,y,w,h = cv2.boundingRect(c)
   print("[INFO] Contour rectangle: {x}, {y}, {w}, {h}".format(x=x, y=y,w=w, h=h))
   if best_box[0] < 0:
       best_box=[x,y,x+w,y+h]
   else:
       if x<best_box[0]:
           best_box[0]=x
       if y<best_box[1]:
           best_box[1]=y
       if x+w>best_box[2]: #and x+w < maxW-10:
           best_box[2]=x+w
       if y+h>best_box[3]: #and y+h < maxH-10:
           best_box[3]=y+h

x,y,w,h = best_box

print("[INFO] Final contour rectangle: {x}, {y}, {w}, {h}".format(x=x, y=y,w=w, h=h))

# Crop rotated, and thresh rotated
crop = rotated[y:h,x:w]

#cropThresh = thresh[y:h,x:w]

# Show the output image
# cv2.imshow("Final", crop)

# Save image
cv2.imwrite(imageNameOut,crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])

#cv2.imwrite(imageNameNegativeOut,cropThresh)
