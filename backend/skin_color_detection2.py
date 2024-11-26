import cv2
import numpy as np

# Load the image
image = cv2.imread('./backend/static/image.png')
def skin_detector(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise AND the skin mask with the original image
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Calculate the average brightness of the skin areas (using the mask)
    skin_pixels = skin[skin_mask == 255]  # Extract pixels where skin is detected

    if len(skin_pixels) > 0:
        # Calculate the average brightness (V channel in HSV represents brightness)
        average_brightness = np.mean(skin_pixels[:, 2])  # V channel is at index 2

        # Classify skin tone based on average brightness
        if average_brightness < 100:
            skin_type = 'Dark'
        elif 100 <= average_brightness < 185:
            skin_type = 'Medium'
        else:
            skin_type = 'Fair'
        
        print(f"Detected Skin Type: {skin_type}")
    else:
        print("No skin detected in the image")
    cv2.imshow("Detected Skin", skin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(skin_type)
    
print(skin_detector(image))