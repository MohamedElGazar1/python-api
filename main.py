import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import pyttsx3
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

engine = pyttsx3.init()
mtcnn_detector = MTCNN()

app = Flask(__name__)

# Define the project directory path
project_dir = "C:/Users/mn/PycharmProjects/testing/"

# Load the TFLite model
model_path = os.path.join(project_dir, 'gender_classification_model_small.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

uploaded_image = None


def preprocess_image(base64_img, target_size=(50, 50)):
    try:
        if base64_img is None:
            raise ValueError("Image data is None")

        image_data = base64.b64decode(base64_img)
        img_np = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image data")

        if img.size == 0:
            raise ValueError("Empty image data")

        cv2.imwrite('uploaded_image.jpg', img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        print("Error preprocessing image:", str(e))
        return None


def detect_faces(image):
    try:
        faces = mtcnn_detector.detect_faces(image)
        faces_rect = [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]
        return faces_rect
    except Exception as e:
        print("Error detecting faces:", str(e))
        return None


def predict_gender(img, faces, threshold=0.5):
    try:
        if faces is None:
            print("No faces detected in the image.")
            return

        gender_predictions = []

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            # Convert input data type to FLOAT32
            face_img = face_img.astype(np.float32)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], face_img)

            # Run inference
            interpreter.invoke()

            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = 1 if output_data[0][0] > threshold else 0
            gender_label = "Male" if predicted_class == 1 else "Female"
            print("Predicted gender for face:", gender_label)
            gender_predictions.append(gender_label)

        return gender_predictions

    except Exception as e:
        print("Error predicting gender:", str(e))
        return None


def speak_results(num_faces, gender_predictions):
    try:
        if num_faces is None or gender_predictions is None:
            return

        engine.say("Number of faces detected: {}".format(num_faces))

        for i, prediction in enumerate(gender_predictions):
            engine.say("Predicted gender for face {}: {}".format(i + 1, prediction))

        engine.runAndWait()

    except Exception as e:
        print("Error speaking results:", str(e))


@app.route('/api', methods=['PUT'])
def index():
    global uploaded_image
    try:
        input_data = request.get_data()
        print("Received data:", input_data)  # Debugging print statement
        uploaded_image = preprocess_image(input_data)
        if uploaded_image is None:
            return 'Failed to preprocess uploaded image', 400
        return 'Image uploaded successfully'
    except Exception as e:
        return str(e), 500


@app.route('/predict', methods=['GET'])
def predict():
    try:
        global uploaded_image
        if uploaded_image is None:
            return 'No image uploaded', 400
        img = cv2.imread('uploaded_image.jpg')
        faces = detect_faces(img)
        gender_predictions = predict_gender(img, faces)
        if gender_predictions:
            speak_results(len(faces), gender_predictions)
            return jsonify({'gender_predictions': gender_predictions})
        else:
            return 'No faces detected or error in gender prediction', 400
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
