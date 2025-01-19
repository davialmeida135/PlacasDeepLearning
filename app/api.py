# FILE: API.PY
import os
from flask import Flask, request, jsonify
from pipeline import LicensePlateRecognition
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import json
import time
import io
from sort_placas import sort_license_plate, time_counter_decorator

app = Flask(__name__)

# Allowed extensions for uploaded files (if you still want to support file uploads)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the LicensePlateRecognition once to avoid reloading models on each request
lpr = LicensePlateRecognition()

@time_counter_decorator
@app.route('/api/v1/license-plate', methods=['POST'])
def license_plate():
    """
    Endpoint to handle license plate recognition.
    """
    start = time.time()
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Open the image file
    image = Image.open(io.BytesIO(file.read()))

    try:
        # Run the license plate recognition pipeline
        result = lpr.run_pipeline(image)

        # Format the result as needed. Assuming 'result' is a list of detections.
        license_plates = [
            {
                'plate': detection[0],
                'bounding_box': {
                    'x1': detection[1],
                    'y1': detection[2],
                    'x2': detection[3],
                    'y2': detection[4]
                }
            }
            for detection in result if detection[0]  # Filter out None results
        ]
        end = time.time()
        total_time = end - start
        # Return the results as JSON
        results = jsonify({'license_plates': license_plates, 'time': total_time})
        print(results)
        return results, 200
    except Exception as e:
        # Log the exception (optional)
        # app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': f'An error occurred while processing the image: {str(e)}'}), 500

if __name__ == '__main__':
    # It's better to set debug to False in production
    app.run( debug=False)