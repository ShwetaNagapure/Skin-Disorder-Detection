import os
import io
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
from flask_sqlalchemy import SQLAlchemy
# Make sure you install weasyprint: pip install weasyprint

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model input size
SIZE = 128

# Load model and label encoder
model = load_model('model/final3.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Prediction Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    infection_area = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<Prediction {self.name} ({self.age}) - {self.predicted_class} - {self.infection_area}>'

# Image Prediction Function
def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((SIZE, SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = le.inverse_transform([predicted_class_index])[0]
    accuracy = prediction[0][predicted_class_index] * 100 

    return predicted_class_label, accuracy

# Mapping labels to full names and malignancy info
full_label_map = {
    'nv': 'Melanocytic Nevi (Benign)',
    'mel': 'Melanoma (Malignant)',
    'bkl': 'Benign Keratosis-like Lesions (Benign)',
    'bcc': 'Basal Cell Carcinoma (Malignant)',
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma (Malignant)',
    'vasc': 'Vascular Lesions (Benign)',
    'df': 'Dermatofibroma (Benign)'
}

# Routes
@app.route('/', methods=['GET'])
def upload_form():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return redirect(url_for('upload_form'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_form'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Get user inputs
    name = request.form.get('name')
    age = int(request.form.get('age'))
    infection_area = request.form.get('infection_area')
    
    predicted_class_label, accuracy = predict_image(filepath)
    full_name = full_label_map.get(predicted_class_label, predicted_class_label)
    malignancy = 'Malignant' if 'Malignant' in full_name else 'Benign'
    
    prediction_text = (f"{name}, Age: {age}, Infection Area: {infection_area} — "
                       f"{full_name} (Accuracy: {accuracy:.2f}%) — {malignancy}")

    # Save to DB
    new_prediction = Prediction(
        filename=filename,
        predicted_class=predicted_class_label,
        confidence=float(accuracy),
        name=name,
        age=age,
        infection_area=infection_area
    )
    db.session.add(new_prediction)
    db.session.commit()

    return render_template(
    'result.html',
    filename=filename,
    name=name,
    age=age,
    infection_area=infection_area,
    predicted_class=full_name, 
    malignancy=malignancy,                
    Accuracy=f"{accuracy:.2f}"
)

# Run app & initialize DB
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

