# import tensorflow as tf
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models

# # Function to convert image to grayscale
# def convert_to_grayscale(image_path):
#     img = cv2.imread(image_path)  # Read image
#     if img is None:
#         raise ValueError(f"Image at path {image_path} could not be loaded. Please check the file path.")
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     return gray_image

# # Function to run pose estimation using MoveNet
# def run_pose_estimation(image_path):
#     # Load MoveNet model from TensorFlow Hub
#     model = tf.saved_model.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     image_resized = cv2.resize(image_rgb, (192, 192))  # Resize to model input size (192x192)
#     image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)
#     image_tensor = image_tensor[tf.newaxis, ...] / 255.0  # Normalize the image

#     # Run the model
#     outputs = model(image_tensor)
#     keypoints = outputs['output_0'].numpy()

#     # Plot the keypoints on the image
#     plt.imshow(image_rgb)
#     for i in range(17):  # 17 keypoints for human body
#         y, x, _ = keypoints[0, i]
#         plt.scatter(x * 192, y * 192, color='red')  # Scale keypoints back to original size
#     plt.show()

# # Function to build a simple CNN model for body shape classification
# def build_model():
#     model = models.Sequential([
#         layers.InputLayer(input_shape=(224, 224, 3)),  # Input image size 224x224
#         layers.Conv2D(32, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(1, activation='sigmoid')  # Binary classification: Male (0), Female (1)
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Function to train the model on dataset
# def train_model(train_images, train_labels):
#     model = build_model()
#     model.fit(train_images, train_labels, epochs=10, batch_size=32)  # Train the model
#     return model

# # Function to preprocess image for model input
# def preprocess_image(image_path):
    
#     image_resized = cv2.resize(image, (224, 224))  # Resize to 224x224 for classification
#     image_normalized = image_resized / 255.0  # Normalize the image
#     return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# # Function to predict body shape using the trained model
# def predict_body_shape(model, image_path):
#     image = preprocess_image(image_path)
#     prediction = model.predict(image)
#     if prediction < 0.6:
#         return "Male"  # Classification output 0 -> Male
#     else:
#         return "Female"  # Classification output 1 -> Female

# # Main execution
# def main(image_path):
#     # Step 1: Convert to Grayscale for better boundary detection
#     gray_image = convert_to_grayscale(image_path)  # Don't overwrite `image_path`
#     cv2.imwrite('grayscale_image.png', gray_image)  # Save grayscale image

#     # Step 2: Run Pose Estimation (PoseNet/MoveNet)
#     print("Running pose estimation...")
#     run_pose_estimation(image_path)

#     # Step 3: Load and use the pre-trained body shape classification model
#     print("Loading body shape classification model...")
#     # Assuming you have already trained your model and saved it
#     model = build_model()  # You would typically load a trained model here
#     # Assuming you have a trained model on 'train_images' and 'train_labels'
#     # For demo, we assume the model is already trained

#     # Step 4: Predict body shape
#     print("Predicting body shape...")
#     body_shape = predict_body_shape(model, image_path)
#     print(f"Predicted Body Shape: {body_shape}")

# # Example usage (image path)image_path = 'C:/Users/KeshavG/OneDrive/Desktop/chic-chameleon/backend/static/image.png'  # Full path to image
# image_path = '.backend/static/image.png'
# image = cv2.imread(image_path)  # Full path to image
#   # Update with your image path
# main(image_path)
# import cv2
# import mediapipe as mp
# import numpy as np
#
# # Initialize MediaPipe Pose model
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
#
# # Predefined Body Type Classifications
# def classify_body_type(gender, shoulder_width, waist_width, hip_width):
#     if gender == "Female":
#         if shoulder_width < waist_width < hip_width:
#             return "Hourglass"
#         elif shoulder_width < waist_width > hip_width:
#             return "Inverted Triangle"
#         elif shoulder_width > waist_width and waist_width < hip_width:
#             return "Rectangle"
#         else:
#             return "Pear"
#     else:
#         if shoulder_width > waist_width and hip_width > waist_width:
#             return "Trapezoid"
#         elif shoulder_width == hip_width:
#             return "Rectangle"
#         else:
#             return "Oval"
#
# # Function to detect pose using MediaPipe and get body landmarks
# def detect_pose(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)
#
#     if results.pose_landmarks:
#         return results.pose_landmarks
#     return None
#
# # Function to calculate measurements based on keypoints (shoulders and hips)
# def calculate_measurements(landmarks):
#     # Get keypoints for shoulders, hips
#     left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#     left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
#     right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
#
#     # Calculate shoulder width (horizontal distance between left and right shoulder)
#     shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2)
#
#     # Calculate hip width (horizontal distance between left and right hip)
#     hip_width = np.sqrt((left_hip.x - right_hip.x)**2 + (left_hip.y - right_hip.y)**2)
#
#     # Use shoulder width as a proxy for waist width (since waist is not directly available)
#     waist_width = shoulder_width  # Estimating waist as similar to shoulder width
#
#     return shoulder_width, hip_width, waist_width
#
# # Function to classify body shape based on measurements
# def body_shape_classification(image, gender="Female"):
#     landmarks = detect_pose(image)
#
#     if landmarks:
#         shoulder_width, hip_width, waist_width = calculate_measurements(landmarks)
#         body_type = classify_body_type(gender, shoulder_width, waist_width, hip_width)
#         return body_type
#     else:
#         return "Pose detection failed"
#
# # Function to draw the outline and return segmented body
# def draw_body_outline(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Use Canny edge detection
#     edges = cv2.Canny(blurred, 50, 150)
#
#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Draw contours on original image
#     body_outline_image = image.copy()
#     cv2.drawContours(body_outline_image, contours, -1, (0, 255, 0), 2)
#
#     return body_outline_image
#
# # Function to get manual body measurements and classify body type
# def manual_body_shape_classification():
#     print("Enter your measurements for body type classification:")
#
#     # Get manual input for measurements
#     chest = float(input("Enter chest measurement (in inches): "))
#     waist = float(input("Enter waist measurement (in inches): "))
#     hips = float(input("Enter hip measurement (in inches): "))
#     gender = input("Enter gender (Male/Female): ")
#
#     # Use manual inputs for classification
#     body_type = classify_body_type(gender, chest, waist, hips)
#     print(f"Body Type based on manual input: {body_type}")
#
#     return body_type
#
# # Main function to process the image and get body type and outline
# def process_image(image_path, gender="Female"):
#     image = cv2.imread(image_path)
#
#     # Get the body shape classification
#     body_type = body_shape_classification(image, gender)
#
#     # Get the body outline
#     body_outline_image = draw_body_outline(image)
#
#     # Show the processed image
#     cv2.imshow("Body Outline", body_outline_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return body_type, body_outline_image
#
# # Example usage
# image_path = './backend/static/image.png'  # Update with actual image path
# body_type, body_outline_image = process_image(image_path, gender="Female")
# print(f"Detected Body Type: {body_type}")
#
# # Option to enter manual measurements
# manual_body_type = manual_body_shape_classification()
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Predefined Body Type Classifications
def classify_body_type(gender, shoulder_width, waist_width, hip_width):
    if gender == "Female":
        if shoulder_width < waist_width < hip_width:
            return "Hourglass"
        elif shoulder_width < waist_width > hip_width:
            return "Inverted Triangle"
        elif shoulder_width > waist_width and waist_width < hip_width:
            return "Rectangle"
        elif shoulder_width < hip_width and waist_width < hip_width:
            return "Pear"
        elif shoulder_width == waist_width == hip_width:
            return "Apple"
        elif waist_width > shoulder_width and waist_width > hip_width:
            return "Oval"
        else:
            return "Other"
    else:  # Male Body Types
        if shoulder_width > waist_width and hip_width > waist_width:
            return "Trapezoid"
        elif shoulder_width == hip_width:
            return "Rectangle"
        elif shoulder_width > hip_width and waist_width < hip_width:
            return "Inverted Triangle"
        elif waist_width > shoulder_width and waist_width > hip_width:
            return "Oval"
        elif shoulder_width > waist_width and waist_width == hip_width:
            return "V-Shaped"
        elif waist_width < shoulder_width and waist_width < hip_width:
            return "Apple"
        else:
            return "Other"

# Function to detect pose using MediaPipe and get body landmarks
def detect_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        return results.pose_landmarks
    return None

# Function to calculate measurements based on keypoints (shoulders and hips)
def calculate_measurements(landmarks):
    # Get keypoints for shoulders, hips
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate shoulder width (horizontal distance between left and right shoulder)
    shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)

    # Calculate hip width (horizontal distance between left and right hip)
    hip_width = np.sqrt((left_hip.x - right_hip.x) ** 2 + (left_hip.y - right_hip.y) ** 2)

    # Estimate waist width using shoulder width as a rough proxy (if not available)
    waist_width = shoulder_width  # Estimating waist as similar to shoulder width

    return shoulder_width, hip_width, waist_width

# Function to classify body shape based on measurements
def body_shape_classification(image, gender="Female"):
    landmarks = detect_pose(image)

    if landmarks:
        shoulder_width, hip_width, waist_width = calculate_measurements(landmarks)
        body_type = classify_body_type(gender, shoulder_width, waist_width, hip_width)
        return body_type
    else:
        return "Pose detection failed"

# Function to draw the outline and return segmented body
def draw_body_outline(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    body_outline_image = image.copy()
    cv2.drawContours(body_outline_image, contours, -1, (0, 255, 0), 2)

    return body_outline_image

# Function to get manual body measurements and classify body type
def manual_body_shape_classification():
    print("Enter your measurements for body type classification:")

    # Get manual input for measurements
    gender = "female"
    if gender.lower() == 'female':
        chest = float(input("Enter bust measurement (in inches): "))
    else:
        chest = float(input("Enter chest measurement (in inches): "))
    waist = float(input("Enter waist measurement (in inches): "))
    hips = float(input("Enter hip measurement (in inches): "))

    # Use manual inputs for classification
    body_type = classify_body_type(gender, chest, waist, hips)
    print(f"Body Type based on manual input: {body_type}")

    return body_type

# Main function to process the image and get body type and outline
def process_image(image_path, gender="Female"):
    image = cv2.imread(image_path)

    # Get the body shape classification
    body_type = body_shape_classification(image, gender)

    # Get the body outline
    body_outline_image = draw_body_outline(image)

    # Show the processed image
    # cv2.imshow("Body Outline", body_outline_image)
    # cv2.waitKey(0)33
    
    # cv2.destroyAllWindows()

    return body_type

# Example usage
image_path = './static/image.png'  # Update with actual image path
gender = "female"
body_type = process_image(image_path, gender="Female")
print(f"Detected Body Type: {body_type}")

# Option to enter manual measurements
body_type = manual_body_shape_classification()








