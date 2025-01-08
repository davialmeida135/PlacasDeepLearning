# Reconhecimento de Placas Veiculares

## Visão Geral

Este projeto utiliza modelos de aprendizado de máquina baseados no [YOLOv11](caracteres/yolo11n.pt) para realizar o reconhecimento e a detecção de placas veiculares. O repositório está organizado em duas principais pastas: `caracteres` e `placas`, cada uma contendo seus próprios conjuntos de dados, scripts de treinamento e resultados de execução.

## Estrutura do Projeto
. 
├── .gitattributes 
├── .gitignore 
├── caracteres/ 
│ ├── dataset/ 
│ │ ├── data.yaml
│ │ ├── README.dataset.txt 
│ │ ├── README.roboflow.txt 
│ │ ├── test/ 
│ │ │ ├── images/ 
│ │ │ ├── labels/ 
│ │ │ └── output/ 
│ │ ├── train/ 
│ │ │ ├── images/
│ │ │ ├── labels/ 
│ │ │ └── labels.cache 
│ │ └── valid/ 
│ │ ├── images/ 
│ │ ├── labels/ 
│ │ └── labels.cache 
│ ├── retrain.py 
│ ├── runs/ 
│ │ └── detect/ 
│ │ └── train/ 
│ ├── test.py 
│ ├── train.py 
│ └── yolo11n.pt 
├── datasets.txt 
├── placas/ 
│ ├── dataset/ 
│ │ ├── data.yaml 
│ │ ├── README.dataset.txt 
│ │ ├── README.roboflow.txt 
│ │ ├── test/ 
│ │ ├── train/ 
│ │ └── valid/ 
│ ├── retrain.py 
│ ├── runs/ 
│ │ └── detect/ 
│ ├── test.py 
│ ├── train.py 
│ └── yolo11n.pt 
└── yolo11n.pt


## Dataset

Os conjuntos de dados para treinamento e validação estão localizados nas pastas `caracteres/dataset` e `placas/dataset`. Cada conjunto de dados inclui imagens e rótulos anotados no formato YOLOv11.

- **Caracteres**: Contém 9.360 imagens com caracteres de placas anotados.
- **Placas**: Contém 9.551 imagens de placas veiculares anotadas.

### Arquivos Importantes

- [`caracteres/dataset/data.yaml`](caracteres/dataset/data.yaml): Configuração do dataset para caracteres.
- [`placas/dataset/data.yaml`](placas/dataset/data.yaml): Configuração do dataset para placas.
- [`caracteres/dataset/README.roboflow.txt`](caracteres/dataset/README.roboflow.txt): Informações sobre o dataset de caracteres.
- [`placas/dataset/README.roboflow.txt`](placas/dataset/README.roboflow.txt): Informações sobre o dataset de placas.

## Treinamento

Os scripts de treinamento estão localizados nas pastas `caracteres` e `placas`.

- **Caractères**:
  - [`caracteres/train.py`](caracteres/train.py): Script principal para treinamento do modelo de caracteres.
  - [`caracteres/retrain.py`](caracteres/retrain.py): Script para retreinamento do modelo.
  
- **Placas**:
  - [`placas/train.py`](placas/train.py): Script principal para treinamento do modelo de placas.
  - [`placas/retrain.py`](placas/retrain.py): Script para retreinamento do modelo.

### Executando o Treinamento

Para treinar o modelo, execute o seguinte comando no terminal integrado do Visual Studio Code:

```sh
python caracteres/train.py

Ou, para o modelo de placas:

python placas/train.py
```
## Resultados
Os resultados do treinamento são salvos nas pastas runs/detect/, contendo os pesos treinados e logs de treinamento.

args.yaml: Configurações do último treinamento.

## Teste

Os scripts de teste estão localizados nas pastas caracteres e placas.

### Caracteres:

test.py: Script para testar o modelo de caracteres.
Placas:


### Placas: 
test.py: Script para testar o modelo de placas.

## TODO

- Diferenciação placa de moto e de carro
- Aplicação modelos de placa e caracteres em sequência
- Documentação do tempo de execução para as detecções


## Requisitos

Python 3.8+
Biblioteca Ultralytics YOLO