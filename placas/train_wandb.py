# filepath: /c:/Users/Davi/Documents/GitHub/DeteccaoPlacas/training-test-yolo/train.py
import os
from ultralytics import YOLO
import wandb
import pathlib
# Inicializar Weights and Biases
wandb.init(project="deteccao-placas")

def main():
    # Carregar um modelo
    model = YOLO("runs/detect/train/weights/best.pt")  # construir um novo modelo do zero
    self_path = pathlib.Path(__file__).parent.resolve()

    # Configurar hiperparâmetros
    hyperparameters = {
        "epochs": 4,
        "batch_size": 16,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "CrossEntropy",
        "activation_function": "ReLU",
        "data_augmentation": True
    }
    wandb.config.update(hyperparameters)
    
    # Usar o modelo
    results = model.train(
        data = os.path.join(self_path, "dataset/data.yaml"),
        epochs=hyperparameters["epochs"],
        batch=hyperparameters["batch_size"],
        lr0=hyperparameters["learning_rate"],
        #optimizer=hyperparameters["optimizer"],
        #loss=hyperparameters["loss_function"],
        #augment=hyperparameters["data_augmentation"]
    )
    
    # Registrar métricas no Weights and Biases
    wandb.log({"accuracy": results.metrics.accuracy, "loss": results.metrics.loss})
    # Finalizar a execução do wandb
    wandb.finish()
    
if __name__ == '__main__':
    main()