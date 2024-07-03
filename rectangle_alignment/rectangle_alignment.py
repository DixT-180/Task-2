import cv2
import numpy as np


image = cv2.imread('rectangles.png')
height, width = image.shape[:2]  

#  image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# edge detection 
#  Canny edge detection
edges = cv2.Canny(gray, 50, 150)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

counter = 0


for contour in contours:
   
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx = approx.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)]
    rect[2] = approx[np.argmax(s)]
    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)]
    rect[3] = approx[np.argmax(diff)]

  
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    filename = f'straightened_image_{counter}.jpg'
    cv2.imwrite(filename, warped)
    print(f'Saved {filename}')
    counter += 1
    cv2.imshow("Original Image", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("Straightened Image", warped)
    cv2.waitKey(0)

cv2.destroyAllWindows()
