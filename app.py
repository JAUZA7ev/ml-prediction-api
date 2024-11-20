import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename
from google.cloud import firestore
from utils import load_model_from_gcs, preprocess_image
from config import BUCKET_NAME, MODEL_PATH

# Initialize Flask app
app = Flask(__name__)

# Initialize Firestore client
db = firestore.Client()

# Constants
MAX_FILE_SIZE = 1_000_000  # 1 MB

# Load model from GCS
model = load_model_from_gcs(BUCKET_NAME, MODEL_PATH)

def save_to_firestore(prediction_id, result, suggestion, timestamp):
    """Save the prediction result to Firestore."""
    prediction_data = {
        "id": prediction_id,
        "result": result,
        "suggestion": suggestion,
        "createdAt": timestamp
    }
    db.collection('predictions').document(prediction_id).set(prediction_data)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'image' not in request.files:
        return jsonify({"status": "fail", "message": "No file part in the request"}), 400

    file = request.files['image']
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    if file_length > MAX_FILE_SIZE:
        return jsonify({"status": "fail", "message": f"Payload content length greater than maximum allowed: {MAX_FILE_SIZE}"}), 413
    
    # Reset file pointer
    file.seek(0)
    filename = secure_filename(file.filename)

    try:
        # Preprocess the image
        image = preprocess_image(file)

        # Perform prediction
        predictions = model.predict(image)
        result = "Cancer" if predictions[0] > 0.5 else "Non-cancer"
        suggestion = "Segera periksa ke dokter!" if result == "Cancer" else "Penyakit kanker tidak terdeteksi."

        # Create unique ID and timestamp
        prediction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Save prediction to Firestore
        save_to_firestore(prediction_id, result, suggestion, timestamp)

        response = {
            "status": "success",
            "message": "Model is predicted successfully",
            "data": {
                "id": prediction_id,
                "result": result,
                "suggestion": suggestion,
                "createdAt": timestamp
            }
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"status": "fail", "message": f"Error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
