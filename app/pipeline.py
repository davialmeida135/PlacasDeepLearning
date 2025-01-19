import os
from ultralytics import YOLO
import cv2
import logging
from sort_placas import sort_license_plate, time_counter_decorator
import pandas as pd
import numpy as np

# Configure logging to suppress verbose output
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class LicensePlateRecognition():
    def __init__(self):
        self.frame_no = 0
        self.char_model = None
        script_dir = os.path.dirname(__file__)

        plate_model_path = os.path.join(script_dir, "../placas/runs/detect/train2/weights/best.pt")
        self.plate_model = YOLO(plate_model_path)

        char_model_path = os.path.join(script_dir, "../caracteres/runs/detect/train3/weights/best.pt")
        self.char_model = YOLO(char_model_path)

        print(self.char_model.info( verbose=True))
        print('====================================================================')
        print(self.plate_model.info(verbose=True))
        
    @time_counter_decorator
    def detect_license_plate(self, frame):

        results = self.plate_model.predict(frame,conf=0.15)

        return results
    
    @time_counter_decorator
    def read_license_plate(self, cropped_img):
        results = self.char_model.predict(cropped_img,conf=0.40)
        detections = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes = result.boxes  
                sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0][0])
                for box in sorted_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    confidence = box.conf[0]
                    class_id = box.cls[0]
                    label = f'{self.char_model.names[int(class_id)]}'
                    detections.append((x1, y1, x2, y2, label, confidence.item()))
            else:
                print("No bounding boxes found in the result for image")

        return detections

    def detect_on_frame(self, frame):
        try:
            print("Detecting license plates...")
            detected_plates = []
            predictions = self.detect_license_plate(frame)
            detections = []
            for pred in predictions:
                if hasattr(pred, 'boxes'):
                    boxes = pred.boxes  
                    sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0][0])
                    for box in sorted_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  
                        cropped_img = frame[y1:y2, x1:x2]
                        #cropped_img = cv2.resize(cropped_img, (640, 640))
                        characters = self.read_license_plate(cropped_img)
                        detections.append((x1, y1, x2, y2, characters))
            

            results = []
            for detection in detections:
                #print(detection)
                x1,y1,x2,y2,lpr = detection
                characters = [char for _, _, _, _, char, _ in lpr]
                bounding_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in lpr]
                confidences = [conf for _, _, _, _, _, conf in lpr]
                #print("Characters: ", characters)
                #print("Boxes: ",bounding_boxes)
                detected_plate = sort_license_plate(bounding_boxes, characters, confidences, plate_coords=(x1, y1, x2, y2))
                
                results.append([detected_plate, x1, y1, x2, y2])
                print("Plate: ", detected_plate)

            return results
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []
        
    def draw_bounding_boxes(self, frame, results):
        for detected_plate, x1, y1, x2, y2 in results:
            if detected_plate:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, detected_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
        return frame

    def run_pipeline(self, image, display:bool = False):
        image = np.array(image) # Convert the image to a numpy array
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)   # Transpose the image
        #image = np.expand_dims(image, axis=0)   # Add a batch dimension
        #image = image / 255.0  # Normalize the image

        results = self.detect_on_frame(image)
        for detected_plate, x1, y1, x2, y2 in results:
            if detected_plate:
                # Desenha a caixa delimitadora
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Adiciona o rótulo
                cv2.putText(image, detected_plate, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 2)
                if display:
                    cv2.imshow('Detected Plate', image)
                    cv2.waitKey(0)

        # Convert the frame color from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'saved/detected_plate{self.frame_no}.jpg', image)
                # Salva a imagem com a placa detectada
        if display:
            cv2.imshow('Detected Plate', image)
            cv2.waitKey(0)

        self.frame_no += 1
        return results
        
if __name__ == "__main__":
    lpr = LicensePlateRecognition()
