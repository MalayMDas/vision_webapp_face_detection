from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from werkzeug.utils import secure_filename
from PIL import Image
import shutil  # For cleaning directories

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# Ensure output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load YOLOv8-Face model
model = YOLO('yolov8l-face.pt')  # Replace with the actual weights path

def cleanup_folders():
    """Deletes all files in the upload and output folders."""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                os.unlink(file_path)  # Delete the file
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Cleanup folders at the start of each run
cleanup_folders()

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded")

    # Perform inference
    results = model.predict(source=image_path, conf=0.5)

    # Extract bounding boxes and confidence
    bboxes = results[0].boxes.xyxy  # Bounding box coordinates
    confidences = results[0].boxes.conf  # Confidence scores

    faces = []
    annotated_image = image.copy()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        confidence = confidences[i]

        # Crop the face from the image
        face = image[y1:y2, x1:x2]
        faces.append(face)

        # Analyze the face using DeepFace
        try:
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]
            dominant_emotion = analysis.get('dominant_emotion', 'Unknown')
            emotions = analysis.get('emotion', {})
            dominant_emotion_score = emotions.get(dominant_emotion, 0) if emotions else 0

            # Draw bounding box and annotations on the original image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{dominant_emotion} ({dominant_emotion_score:.2f}%)"
            y_offset = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(annotated_image, label, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")

    # Save annotated image
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_image.jpg')
    cv2.imwrite(output_image_path, annotated_image)

    # Save cropped faces
    face_paths = []
    for idx, face in enumerate(faces):
        face_path = os.path.join(app.config['OUTPUT_FOLDER'], f'face_{idx}.jpg')
        cv2.imwrite(face_path, face)
        face_paths.append(face_path)

    return output_image_path, face_paths

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Process the uploaded image
            try:
                annotated_image, face_paths = process_image(upload_path)
                return render_template('result.html', annotated_image=annotated_image, face_paths=face_paths)
            except Exception as e:
                return f"Error processing image: {e}"

    return render_template('upload.html')

@app.route('/delete', methods=['POST'])
def delete_files():
    cleanup_folders()
    return redirect(url_for('upload_file'))

@app.route('/delete_from_results', methods=['POST'])
def delete_files_from_results():
    cleanup_folders()
    return redirect(url_for('upload_file'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

