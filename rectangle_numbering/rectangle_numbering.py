
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image = cv2.imread('rectangles.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (example using Canny edge detector)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Find contours with RETR_TREE to get all contours including internal ones
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to exclude very small contours
min_area = 100  # Adjust as necessary
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Function to check if a contour is a rectangle
def is_rectangle(contour, epsilon=0.02):
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    # A rectangle has 4 vertices
    return len(approx) == 4

# Filter out rectangular contours
non_rectangular_contours = [cnt for cnt in filtered_contours if not is_rectangle(cnt)]

# Calculate lengths of all non-rectangular contours
contour_lengths = [(cv2.arcLength(cnt, closed=False), cnt) for cnt in non_rectangular_contours]

# Sort contours by length in increasing order
contour_lengths.sort(key=lambda x: x[0])

# Print the initial lengths of all non-rectangular contours
print("Initial lengths of non-rectangular contours (sorted):")
for length, _ in contour_lengths:
    print(length)

# Function to calculate the midpoint of a contour
def contour_midpoint(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return np.array([cx, cy])

# Calculate distances between all contour midpoints
midpoints = [contour_midpoint(cnt) for _, cnt in contour_lengths]
merged_contours = []
used = [False] * len(contour_lengths)

# Merge contours that are closer than the specified threshold
threshold = 2.70

for i in range(len(contour_lengths)):
    if used[i]:
        continue
    length_sum = contour_lengths[i][0]
    count = 1
    for j in range(i + 1, len(contour_lengths)):
        if used[j]:
            continue
        distance = np.linalg.norm(midpoints[i] - midpoints[j])
        if distance < threshold:
            length_sum += contour_lengths[j][0]
            count += 1
            used[j] = True
    merged_length = length_sum / count
    merged_contours.append((merged_length, contour_lengths[i][1]))
    used[i] = True

# Sort the merged contours by length again
merged_contours.sort(key=lambda x: x[0])

# Print the lengths of merged non-rectangular contours
print("\nLengths of merged non-rectangular contours (sorted):")
for length, _ in merged_contours:
    print(length)

# Draw and label the first 4 merged non-rectangular contours
font_scale = 1.5
thickness = 2
for rank, (length, contour) in enumerate(merged_contours[:4]):
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    cv2.putText(image, f'{rank + 1}', (contour[0][0][0], contour[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

# Draw all remaining merged non-rectangular contours
for length, contour in merged_contours[4:]:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Show the image with all detected contours and lengths
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.savefig('rectangles_numbered_1_4.png') 
plt.title('Ordered Numerically')
plt.axis('off')
plt.show()
 
