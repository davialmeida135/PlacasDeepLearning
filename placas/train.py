import os
import pathlib
from ultralytics import YOLO
#LEVOU UMA HORA
def get_last_model():
    self_path = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(self_path, "runs/detect/train2/weights/best.pt")
    return model_path

def main():
    # Load a model
    model = YOLO(get_last_model())  # Load a model from a file
    #model = YOLO("yolo11n.yaml")  # build a new model from scratch
    self_path = pathlib.Path(__file__).parent.resolve()
    data_path = os.path.join(self_path, "dataset2/data.yaml")
    # Use the model
    results = model.train(data=data_path, epochs=15)  # train the model

if __name__ == "__main__":
    main()