from flask import Flask, render_template, request, url_for, redirect
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CHANGE 1: Point UPLOAD_FOLDER to be inside the 'static' directory ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your trained model
model = load_model('rice.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Map the model's output indices to the actual rice type names
label_map = {0: "Arborio", 1: "Basmati", 2: "Ipsala", 3: "Jasmine", 4: "Karacadag"}

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/details')
def details():
    """Renders the page for uploading an image."""
    return render_template('details.html')

@app.route('/results', methods=['POST'])
def results():
    """Handles the file upload, runs prediction, and shows the result."""
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_label = label_map[predicted_class_index]

        return render_template('results.html', label=predicted_label, image_file=filename)

@app.route('/results', methods=['GET'])
def results_get():
    return redirect('/details')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
