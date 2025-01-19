import os
import cv2
import requests
import base64
import json
import numpy as np

def encode_frame_to_base64(frame):
    """Encode a frame (NumPy array) to a Base64 string."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError("Could not encode frame to JPEG")
    
    return buffer.tobytes()

def send_frame_to_api(image, url='http://127.0.0.1:5000/api/v1/license-plate'):
    """Send Base64-encoded image to the Flask API."""
    files = {'image': ('frame.jpg', image, 'image/jpeg')}

    response = requests.post(url, files=files)
    return response

def main(video_path):
    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    frame_count = 0

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Encode frame to Base64
        encoded_img = encode_frame_to_base64(frame)

        response = send_frame_to_api(encoded_img)

        # Send the encoded image to the API
        #response = send_frame_to_api(base64_image)
        #print(f'Response for frame {frame_count}: {response.text}')
        print('ola')
        print(response.text)

        frame_count += 1
        # Display the frame with bounding boxes
        cv2.imshow('Annotated Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = 'test_placa.jpg'  # Replace with your video file path
    main(video_file)