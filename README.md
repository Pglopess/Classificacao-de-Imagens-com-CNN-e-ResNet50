# Classificacao de Imagens com CNN e MobileNetV2

Experimento de classificacao de imagens com redes neurais convolucionais em Python, comparando arquiteturas CNN customizadas com transfer learning via MobileNetV2.

## Sobre

O projeto treina e compara CNNs customizadas com um modelo MobileNetV2 pre-treinado no ImageNet sobre um dataset de 50 modelos de tenis (~6.500 imagens). As imagens originais possuem resolucao media de 140x125px, o que motivou o uso de input size 96x96 em vez do padrao 224x224, evitando upscaling sem informacao real. O pipeline inclui augmentation, early stopping e fine-tuning em duas fases. Apos o treino, um script de inferencia permite classificar novas imagens via linha de comando.

## Funcionalidades

- Extracao automatica do dataset a partir de um `.zip`
- Data augmentation no treino (rotacao, zoom, flip, brilho)
- CNN customizavel com BatchNormalization e Dropout
- Teste de 3 configuracoes distintas de CNN
- Transfer learning com MobileNetV2 em duas fases: head isolado e fine-tuning das ultimas camadas
- EarlyStopping, ModelCheckpoint e ReduceLROnPlateau em todos os experimentos
- Salvamento automatico do melhor modelo por val_accuracy
- Exportacao de curvas de treino/validacao em PNG
- Tabela comparativa de resultados salva em CSV

## Modelos comparados

| Modelo | Conv layers | Filtros iniciais | Otimizador | Fine-tuning |
|--------|-------------|------------------|------------|-------------|
| CNN 1  | 2           | 32               | Adam       | nao         |
| CNN 2  | 3           | 64               | Adam       | nao         |
| CNN 3  | 4           | 64               | Adam       | nao         |
| MobileNetV2 | -      | -                | Adam       | sim (2 fases) |

## Tecnologias

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib

## Instalacao

```bash
pip install -r requirements.txt
```

## Como executar

### Treino

Coloque o arquivo `dataset_marca.zip` na mesma pasta e execute:

```bash
python sneaker_classifier.py --zip dataset_marca.zip
```

Os resultados serao salvos em `resultados/`.

### Inferencia

```bash
python predict.py --model resultados/MobileNetV2.keras --image foto.jpg
python predict.py --model resultados/MobileNetV2.keras --image foto.jpg --top 5
```

## Estrutura do projeto

```
├── sneaker_classifier.py      # Pipeline completo: dados, modelos e comparacao
├── predict.py                 # Inferencia em imagem unica via CLI
├── requirements.txt           # Dependencias com versoes fixas
├── dataset_marca.zip          # Dataset de imagens (nao versionado)
└── resultados/                # Gerado no treino (nao versionado)
    ├── class_labels.json          # Mapeamento index -> classe
    ├── resultados.csv             # Tabela comparativa de todos os modelos
    ├── CNN_2conv_32f.keras
    ├── CNN_3conv_64f.keras
    ├── CNN_4conv_64f.keras
    ├── MobileNetV2.keras
    └── *_curvas.png               # Curvas de treino e validacao por modelo
```

## Conceitos praticados

- Redes neurais convolucionais (CNN)
- Transfer learning e fine-tuning em duas fases
- Data augmentation com ImageDataGenerator
- BatchNormalization e Dropout
- EarlyStopping e ReduceLROnPlateau
- Inferencia em producao com modelo salvo
- Avaliacao comparativa de arquiteturas
