# Classificação de Imagens com CNN e ResNet50

Experimento de classificação de imagens com redes neurais convolucionais em Python, comparando arquiteturas CNN customizadas com transfer learning via ResNet50.

## Sobre

O projeto treina e compara diferentes configurações de CNN sobre um dataset de imagens de marcas, avaliando acurácia de treino e validação. Ao final, compara os resultados com um modelo ResNet50 pré-treinado no ImageNet.

## Funcionalidades

- Extração automática do dataset a partir de um `.zip`
- Visualização de amostras do dataset
- CNN customizável (número de camadas, filtros e função de ativação)
- Teste de 3 configurações distintas de CNN
- Transfer learning com ResNet50 (pesos ImageNet, camadas congeladas)
- Salvamento automático do melhor modelo CNN (`melhor_modelo.keras`)
- Comparação final de todos os modelos em tabela

## Modelos comparados

| Modelo | Conv layers | Filtros | Ativação | Otimizador |
|--------|-------------|---------|----------|------------|
| CNN 1  | 2           | 32      | relu     | adam       |
| CNN 2  | 3           | 64      | relu     | adam       |
| CNN 3  | 4           | 128     | tanh     | sgd        |
| ResNet50 | -         | -       | relu     | adam       |

## Tecnologias

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib

## Instalação

```bash
pip install tensorflow numpy pandas matplotlib
```

## Como executar

Coloque o arquivo `dataset_marca.zip` na mesma pasta que `main.py`, depois execute:

```bash
python main.py
```

## Estrutura do projeto

```
├── main.py                    # Pipeline completo: dados, modelos e comparação
├── dataset_marca.zip          # Dataset de imagens (não versionado)
├── melhor_modelo.keras        # Melhor CNN salva automaticamente
└── melhor_modelo_resnet.keras # Modelo ResNet50 treinado
```

## Conceitos praticados

- Redes neurais convolucionais (CNN)
- Transfer learning
- Data augmentation e generators
- Comparação de hiperparâmetros
- Avaliação de modelos com acurácia de validação
