import cv2
import numpy as np
from collections import Counter
import glob
import os

# Dynamically find the image in the static folder (handles .jpg, .jpeg, .png, etc.)
image_path = glob.glob('../frontend/assets/image.*')  # Matches any extension like .jpg, .png, .jpeg, etc.

if not image_path:
    raise FileNotFoundError("No image found in the static folder.")

# Load the image
image = cv2.imread(image_path[0])  # Get the first match

if image is None:
    raise ValueError("Failed to load the image. Please check the file format or path.")
# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_skin_color(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjusted skin color range to capture a broader spectrum
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 150, 255], dtype=np.uint8)

    # Create a mask for the skin
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No faces detected in the image"

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the image
        face_region = image[y:y+h, x:x+w]
        face_skin_mask = skin_mask[y:y+h, x:x+w]

        # Apply Gaussian blur to smooth the image and reduce noise
        blurred_face_region = cv2.GaussianBlur(face_region, (5, 5), 0)

        # Extract skin pixels from the face region
        face_skin_pixels = blurred_face_region[face_skin_mask == 255]

        if len(face_skin_pixels) > 0:
            # Get the most common color from the face skin pixels
            face_skin_pixels = [tuple(pixel) for pixel in face_skin_pixels]
            most_common_color = Counter(face_skin_pixels).most_common(1)[0][0]

            # Convert BGR to Hex for display
            skin_color_code = f"#{most_common_color[2]:02x}{most_common_color[1]:02x}{most_common_color[0]:02x}"

            # Create a color block image to show the detected skin color
            color_block = np.zeros((100, 100, 3), dtype=np.uint8)
            color_block[:] = most_common_color  # Fill the block with the detected color

            # Show the original image with the detected face and skin color
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face
            cv2.putText(image, f"Skin Color: {skin_color_code}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show the color block
            # cv2.imshow("Detected Skin Color", color_block)
            # cv2.imshow("Detected Face and Skin Color", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return skin_color_code
        else:
            return "No skin detected on the face region"

# Get and display the skin color from the face region
skin_color_code = get_skin_color(image)
print(f"Detected Skin Color Code: {skin_color_code}")
