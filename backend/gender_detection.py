# import cv2
# import numpy as np
# from keras.preprocessing import image
# from keras.models import load_model

# # # Load your pre-trained gender classification model
# # model = load_model('./backend/models/model.h5')  # Ensure 'model.h5' exists in your working directory

# # # Load the face detection classifier from OpenCV (Haar Cascade)
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Function to detect gender in an image
# def gender_detector(img_path):
#     # Load your pre-trained gender classification model
#     model = load_model('./backend/models/model.h5')  # Ensure 'model.h5' exists in your working directory

#     # Load the face detection classifier from OpenCV (Haar Cascade)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     # Load the input images
#     img = cv2.imread(img_path)
#     if img is None:
#         print("Error: Unable to read the image. Check the image path.")
#         return

#     # Convert image to grayscale for face detection
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

#     if len(faces) == 0:
#         print("No faces detected.")
#         return

#     for (x, y, w, h) in faces:
#         # Extract face region
#         face = img[y:y+h, x:x+w]

#         # Resize and preprocess face image for the model
#         face_resized = cv2.resize(face, (64, 64))  # Resize to match model input size
#         face_array = image.img_to_array(face_resized)
#         face_array = np.expand_dims(face_array, axis=0)
#         face_array = face_array / 255.0  # Normalize input to [0, 1] range

#         # Predict gender
#         gender_prediction = model.predict(face_array)
#         gender = 'Male' if gender_prediction[0][0] > 0.7 else 'Female'

#         # Draw rectangle around the face and display the predicted gender
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


#     # Display the image with gender detection results
#     cv2.imshow('Gender Detection', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return(gender)

# # Run the gender detection function
#   # Replace 'image.png' with the path to your input image
# gender = gender_detector('./backend/static/image.png')
# print(gender)

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Function to detect gender in an image
def gender_detector(img_path):
    # Load your pre-trained gender classification model
    model = load_model('../backend/models/model.h5')  # Ensure 'model.h5' exists in your working directory

    # Load the face detection classifier from OpenCV (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the input image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to read the image. Check the image path.")
        return

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        # Extract face region
        face = img[y:y+h, x:x+w]

        # Resize and preprocess face image for the model
        face_resized = cv2.resize(face, (64, 64))  # Resize to match model input size
        face_array = image.img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = face_array / 255.0  # Normalize input to [0, 1] range

        # Predict gender with confidence feedback
        gender_prediction = model.predict(face_array)
        confidence = gender_prediction[0][0]
        print(f"Raw Prediction Confidence: {confidence:.2f}")

        # Dynamic threshold
        threshold = 0.65  # Default threshold
        gender = 'Male' if confidence > threshold else 'Female'

        # Print predicted gender and confidence
        print(f"Predicted Gender: {gender} (Male Confidence: {confidence:.2f})")

        # Draw rectangle around the face and display the predicted gender
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{gender} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return gender
    # Display the image with gender detection results
    # cv2.imshow('Gender Detection', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Run the gender detection function
# Replace 'image.png' with the path to your input image
gender = gender_detector('../backend/static/image.png')
